alias j=autojump
alias relogin="source ~/.bashrc"
alias vimlogin="vim ~/.bashrc && relogin"
alias vimaliases="vim ~/.bash_aliases && relogin"
alias d=pushd
alias u=popd
alias lrt="ls -lrt"

alias l2server="cd ~/l2race &&  conda activate l2race && python -m server --log=INFO"
alias l2restart="cd ~/l2race && git pull && systemctl --user restart l2race-server.service"
alias l2tail="journalctl --follow --user"
alias l2stop="systemctl --user stop l2race-server.service"
