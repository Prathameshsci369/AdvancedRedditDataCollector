First Command are(ubuntu) sudo apt update

    sudo apt install ansible -y

    ansible --version

    mkdir ~/ansible-lab

    cd ~/ansible-lab

    nano hosts (alternative "mousepad hosts") paste following code on the hosts file localhost ansible_connection = local on the nano press CTRL + O and after press CTRL + X

    after the creating the above file now execute this comments

    '' ansible  -i hosts local -m ping ''

    after the above excutive see the green line

    now create one more file , name

    nano install_nginx.yml

    paste following content

    ""

    -name: Install and start NGINX on localhost

    hosts: local

    become: yes

    tasks:

    -name: Install NGINX

    apt:

    name:nginx

    state: present

    update_cache:yes

    -name: Ensure NGINX is running

    service :

    name: nginx

    state: started

    enabled: yes

    ""

    save the file and execute following command

    ansible-playbook -i hosts install_nginx.yml

    after the this command run the following command;-

    curl http://localhost

