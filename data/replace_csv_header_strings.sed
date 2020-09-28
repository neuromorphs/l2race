#! /usr/bin/sed -f
# replaces old headers with new ones, run with bash command below for hierarchy of folders
# for f in $(find . -name \*.csv); do sed -i -f replace_csv_header_strings.sed $f; done

s/cmd.auto/command.autodrive_enabled/g
s/cmd.steering/command.steering/g
s/cmd.throttle/command.throttle/g
s/cmd.brake/command.brake/g
s/cmd.reverse/command.reverse/g
s/pos.x/position_m.x/g
s/pos.y/position_m.y/g
s/vel.x/velocity_m_per_sec.x/g
s/vel.y/velocity_m_per_sec.y/g
s/speed/speed_m_per_sec/g
s/accel.x/accel_m_per_sec_2.x/g
s/accel.y/accel_m_per_sec_2.y/g
s/steering_angle/steering_angle_deg/g
s/body_angle/body_angle_deg/g
s/yaw_rate/yaw_rate_deg_per_sec/g
s/drift_angle/drift_angle_deg/g