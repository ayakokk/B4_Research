FROM ubuntu:latest

# update
RUN apt update -y 
RUN apt install -y wget git
RUN apt install -y libxrender-dev # rdkit

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/Miniconda3-latest-Linux-x86_64.sh
RUN bash /opt/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
RUN rm -f /opt/Miniconda3-latest-Linux-x86_64.sh
# set path
ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install jupyter numpy pandas scikit-learn seaborn
RUN pip install rdkit pubchempy
RUN pip install xhtml2pdf
RUN pip install autopep8 flake8

RUN echo "mkdir -p /root/.ssh" >> /etc/bash.bashrc
RUN echo "cp /root/.ssh_host/* /root/.ssh/" >> /etc/bash.bashrc
RUN echo "chmod 600 /root/.ssh/*" >> /etc/bash.bashrc