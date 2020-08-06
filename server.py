import argparse
import atexit
import logging
import socket, pickle
from queue import Empty
from typing import Dict, Tuple, List, Optional

import argcomplete
from timeit import default_timer as timer
from time import sleep
import multiprocessing as mp

from src.car_state import car_state
from src.l2race_utils import bind_socket_to_range, set_logging_level, loop_timer, become_daemon, \
    find_unbound_port_in_range
from src.my_args import server_args
from src.car_model import car_model
from src.car import car
from src.globals import *
from src.track import track, list_tracks
from src.l2race_utils import my_logger
logger=my_logger(__name__)
SKIP_CHECK_SERVER_QUEUE=0 # use to reduce checking queue, but causes timeout problems with adding car if too big. 0 to disable

def get_args():
    parser = argparse.ArgumentParser(
        description='l2race client: run this if you are a racer.',
        epilog='Run with no arguments to open dialog for server IP', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = server_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

def send_message(socket:socket, lock:mp.Lock, client_addr: Tuple[str,int], msg:object):
    try:
        logger.debug('sending msg {} to client {}'.format(msg,client_addr))
        p=pickle.dumps(msg)
        if lock: lock.acquire()
        try:
            socket.sendto(p, client_addr)
        except OSError as e:
            logger.error('failed sending msg {} to client {}: {}'.format(msg,client_addr,e))
    finally:
        if lock: lock.release()

class track_server_process(mp.Process):
    def __init__(self, queue_from_server:mp.Queue, server_port_lock:mp.Lock(), server_socket:socket, track_name=None, port: int=None):
        super(track_server_process, self).__init__(name='track_server_process-{}'.format(track_name))
        self.server_queue=queue_from_server
        self.server_port_lock=server_port_lock
        self.server_socket=server_socket # used for initial communication to client who has not yet sent anything to us on the new port
        self.track_name = track_name
        self.track=None  # create after start since Process.spawn cannot pickle it
        self.car_dict:Dict[Tuple[str,int],car_model] = None # maps from client_addr to car_model (or None if a spectator)
        # each client process should bind it's own unique local port (on remote client) so should be unique in dict
        self.car_states_list:List[car_state]=None # list of all car states, to send to clients and put in each car's state
        self.spectator_list:List[Tuple[str,int]] = None # maps from client_addr to car_model (or None if a spectator)
        self.track_socket:Optional[socket] = None # make a new datagram socket
        self.local_port_number=port
        self.track_socket_address=None # get the port info for our local port
        self.exit=False
        self.last_message_time=timer() # used to terminate ourselves if no messages for some time
        self.skip_checking_server_queue_count=0

        atexit.register(self.cleanup)

    def cleanup(self):
        logger.debug('cleaning up {} process'.format(self.track_name))
        if self.car_dict:
            for c in self.car_dict.keys():
                self.send_client_msg(c,'track_shutdown','track server has shut down')
        if self.spectator_list:
            for s in self. spectator_list:
                self.send_client_msg(s,'server_shutdown','track server has shut down')
        # empty queue
        if self.server_queue and not self.server_queue.empty:
            item = self.server_queue.get(block=False)
            while item:
                try:
                    self.server_queue.get(block=False)
                except Empty:
                    break

        self.server_queue.close()
        self.track_socket.close()


    def run(self):
        # logger.setLevel(logging.DEBUG)
        logger.info("Starting track process track {}".format(self.track_name))
        self.track=track(self.track_name)
        self.car_dict = dict() # maps from client_addr to car_model (or None if a spectator)
        self.car_states_list=list() # list of all car states, to send to clients and put in each car's state
        self.spectator_list = list() # maps from client_addr to car_model (or None if a spectator)
        self.track_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # make a new datagram socket
        self.track_socket.settimeout(0) # put track socket in nonblocking mode to just poll for client messages
        # find range of ports we can try to open for client to connect to
        try:
            self.track_socket.bind(('0.0.0.0',self.local_port_number))
        except Exception as e:
            logger.error('track process aborting: could not bind to the local port {} that server told us to use: got {}'.format(self.local_port_number, e))
            raise e
        self.track_socket_address=self.track_socket.getsockname() # get the port info for our local port
        logger.info('for track {} bound free local UDP port address {}'.format(self.track_name,self.local_port_number))
        last_time=timer()

        # Track process makes a single socket bound to a single port for all the clients (cars and spectators).
        # To handle multiple clients, when it gets a message from a client, it responds to the client using the client address.

        looper=loop_timer(MODEL_UPDATE_RATE_HZ)
        while not self.exit:
            now=timer()
            dt=now-last_time
            last_time=now
            if now-self.last_message_time>KILL_ZOMBIE_TRACK_TIMEOUT_S:
                logger.warning('track process {} got no input for {}s, terminating'.format(self.track_name,KILL_ZOMBIE_TRACK_TIMEOUT_S))
                self.exit=True
                self.cleanup()
                continue
            self.process_server_queue() # 'add_car' 'add_spectator'

            # now we do main simulation/response
            # update all the car models
            for client,model in self.car_dict.items():
                if isinstance(model,car_model):
                    model.update(dt)
                # poll for UDP messages
            # update the global list of car states that cars share
            self.car_states_list.clear()
            for model in self.car_dict.values():
                self.car_states_list.append(model.car_state)

            # for each car, fill its car_state.other_car_states with the other cars
            for s in self.car_states_list:
                s.other_car_states.clear()
                for s2 in self.car_states_list:
                    if not s2==s:
                        s.other_car_states.append(s2)

            # process incoming UDP messages from clients, e.g. to update command
            while True:
                try:
                    msg,payload,client=self.receive_msg()
                    self.handle_client_msg(msg, payload, client)
                except socket.timeout:
                    break
                except BlockingIOError:
                    break
                except Exception as e:
                    logger.warning('caught Exception {} while processing UDP messages from client'.format(e))
                    break
            try:
                looper.sleep_leftover_time()
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt, stopping server')
                self.exit=True
                continue


        self.cleanup()
        logger.info('ended track {}'.format(self.track_name))

    def receive_msg(self):
        """ receives a message from client using track's socket"""
        p,client=self.track_socket.recvfrom(2048)
        (msg,payload)=pickle.loads(p)
        logger.debug('got msg={} with payload={} from client {}'.format(msg,payload,client))
        return msg,payload,client

    def send_client_msg(self,client, msg,payload):
        """ sends message back to client using the track's socket"""
        send_message(self.track_socket, None, client, (msg,payload))

    def handle_client_msg(self, msg, payload, client):
        """ handles the client messages """
        logger.debug('handling msg={} with payload={} from client {}'.format(msg,payload,client))
        self.last_message_time=timer()
        # check if spectator or car
        if msg=='command':
            car_model=self.car_dict.get(client)
            if car_model is None:
                logger.warning('car model=None for client {}'.format(client))
                return
            car_model.car_state.command=payload
            msg='car_state'
            payload=car_model.car_state
            self.send_client_msg(client, msg,payload)
        elif msg=='send_states':
            msg='all_states'
            payload=self.car_states_list
            self.send_client_msg(client, msg,payload)
        elif msg=='remove_car':
            car_model=self.car_dict.get(client)
            if not car_model is None:
                logger.info('removing car {} from track {}'.format(car_model.car_name, self.track_name))
                del self.car_dict[client]
        elif msg=='remove_spectator':
            logger.info('removing spectator {} from track {}'.format(client,self.track_name))
            self.spectator_list.remove(client)
        elif msg == 'finish_race':
            logger.info('Removing {} from track'.format(self.car_dict[client].car_name))
            del self.car_dict[client]
            pass
        else:
            logger.warning('unknowm cmd {} received; ignoring'.format(msg))

    def add_car_to_track(self, car_name, client_addr):
        """ adds a car to this track """
        if self.car_dict.get(client_addr):
            logger.warning('client at {} already has a car model, replacing it with a new model'.format(client_addr))
        logger.info('adding car model for car named {} from client {} to track {}'.format(car_name,client_addr,self.track_name))
        mod=car_model(track=self.track, car_name=car_name)
        self.car_dict[client_addr]=mod

    def add_spectator_to_track(self, client_addr):
        """ adds a spectator to this track """
        logger.debug('adding spectator from client {} to track {}'.format(client_addr,self.track_name))
        self.spectator_list.append(client_addr)

    def process_server_queue(self):
        if SKIP_CHECK_SERVER_QUEUE>0:
            self.skip_checking_server_queue_count+=1
            if self.skip_checking_server_queue_count%SKIP_CHECK_SERVER_QUEUE!=0: return
        while not self.server_queue.empty():
            (cmd,payload)=self.server_queue.get_nowait()
            self.handle_server_msg(cmd,payload)
        pass

    def handle_server_msg(self, cmd,payload):
        logger.debug('got queue message from server manager cmd={} payload={}'.format(cmd,payload))
        self.last_message_time=timer()
        if cmd=='stop':
            logger.info('track {} stopping'.format(self.track_name))
            self.cleanup()
            self.exit=True
        elif cmd=='add_car':
            (car_name, client_addr)=payload
            self.add_car_to_track(car_name, client_addr)
        elif cmd=='add_spectator':
            client_addr=payload
            self.add_spectator_to_track(client_addr)
        else:
            raise RuntimeWarning('unknown cmd {}'.format(cmd))

if __name__ == '__main__':
    try:
        from scripts.regsetup import description
        from gooey import Gooey  # pip install Gooey
    except Exception:
        logger.warning('Gooey GUI builder not available, will use command line arguments.\n'
                       'Install with "pip install Gooey". See README')
    try:
        ga = Gooey(get_args, program_name="l2race server", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey"')
    args = get_args()
    set_logging_level(args)


    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('', SERVER_PORT)) # bind to empty host, so we can receive from anyone on this port
    logger.info("waiting on {}".format(str(server_socket)))
    server_port_lock=mp.Lock() # proceeses get passed this lock to initiate connections using it (but only once, at start)

    track_names=list_tracks()
    track_processes:Dict[str,mp.Process] = {k:None for k in track_names} # each entry holds the track objects for each track name
    track_queues:Dict[str,mp.Queue]={k:None for k in track_names} # each entry is the queue to send to track process

    def make_track_process(track_name, client_addr) -> mp.Process:
        track_port_number=find_unbound_port_in_range(CLIENT_PORT_RANGE)
        send_game_port_to_client(client_addr,track_port_number)
        if not track_processes.get(track_name) is None\
                and track_processes.get(track_name).is_alive():
            logger.info('track process {} already exists already and is alive')
            return track_processes.get(track_name)
        logger.info('starting a new track_server_process for track {} for client at {} using local port {}'
                    .format(track_name, client_addr, track_port_number))
        q=mp.Queue()
        track_queues[track_name]=q
        track_process = track_server_process(queue_from_server=q,
                                             server_port_lock=server_port_lock,
                                             server_socket=server_socket,
                                             track_name=track_name,
                                             port=track_port_number)
        track_processes[track_name]=track_process
        track_processes[track_name].start()
        return track_process

    def send_game_port_to_client(client_addr:Tuple[str,int], port:int):
        logger.info('sending game_port message to client {} telling it to use our local port number {}'.format(client_addr,port))
        # first message to client is the game port number
        # send client the port they should use for this track
        send_message(socket=server_socket, lock=server_port_lock,
                     client_addr=client_addr,
                     msg=('game_port',port))

    def add_car_to_track(track_name, car_name, client_addr):
        make_track_process(track_name=track_name, client_addr=client_addr)
        logger.info('putting message to track process for track {} to add car named {} for client {}'.format(track_name,car_name,client_addr))
        q = track_queues.get(track_name)
        if q:
            q.put(('add_car', (car_name, client_addr)))

    def add_spectator_to_track(track_name, client_addr):
        make_track_process(track_name=track_name, client_addr=client_addr)
        q=track_queues[track_name]
        if q:
            q.put(('add_spectator',client_addr))

    def stop_all_track_processes():
        for t,q in track_queues.items():
            if q:
                logger.info('telling track {} to stop'.format(t))
                try:
                    q.put('stop')
                except:
                    pass
        sleep(1)
        logger.info('joining processes')
        for t,p in track_processes.items():
            if p: p.join(1)
        for t,p in track_processes.items():
            if p and p.is_alive():
                logger.info('terminating zombie track process {}'.format(p))
                p.terminate()
        track_processes.clear()
        logger.info('closing queues')
        for q in track_queues.values():
            if q:
                q.close()
                q.join_thread()
        track_queues.clear()

    def cleanup():
        logger.debug('cleaning up server main process')
        stop_all_track_processes()

    atexit.register(cleanup)

    # become_daemon() # todo for service mode

    # We fork an mp process for each track that might have one or more cars and spectators.

    # There is only a single instance of each track. Any clients that want to use that track share it.

    # Each track also gets a Queue which main process uses to tell it when there are new cars for it.

    # Each track runs single threaded to model all the cars in real time, and responds to commands (or spectate state requests) sent from clients with car states
    # of all the cars on that track.

    # Flow is like this:
    # 1. Server waits on SERVER_PORT
    # 2. Client sends newcar to SERVER_PORT
    # 3. Server responds to same port on client with ack and new port
    # 4. Client talks to track process on new port
    # That way, client initiates communication on new port and should be able to recieve on it


     # handling proceeses based on https://www.cloudcity.io/blog/2019/02/27/things-i-wish-they-told-me-about-multiprocessing-in-python/

    while True: # todo add KeyboardInterrupt exception handling, also SIG_TERM
        try:
            server_port_lock.acquire()
            data, client_addr = server_socket.recvfrom(1024)  # buffer size is 1024 bytes
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt, stopping server')
            break
        finally:
            server_port_lock.release()
        try:
            (cmd, payload) = pickle.loads(data)
        except pickle.UnpicklingError as ex:
            logger.warning('{}: garbled command, ignoring. \n'
                           'Client should send pickled 2-tuple (cmd, payload).\n '
                           'cmd="add_car|add_spectator"\n'
                           'payload (for add_car) =(track_name,car_name)\n'
                           'payload (for add_spectator) =(track_name)\n'
                           .format(ex))
            continue

        logger.info('received cmd "{}" with payload "{}" from {}'.format(cmd, payload, client_addr))
        # todo handle multiple cars on one track, provide option for unique track for testing single car

        if cmd=='ping':
            msg=('pong',None)
            send_message(server_socket,server_port_lock,client_addr,msg)
        elif cmd == 'add_car': # todo add arguments with newcar like driver/car name
            (track_name, car_name)=payload
            add_car_to_track(track_name, car_name, client_addr)
        elif cmd=='add_spectator':
            track_name=payload
            add_spectator_to_track(track_name, client_addr)
        else:
            logger.warning('model server received unknown cmd={}'.format(cmd))
