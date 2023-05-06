FROM ubuntu:22.04

RUN apt update && apt -y upgrade
RUN apt -y install wget
RUN wget https://github.com/gohugoio/hugo/releases/download/v0.91.2/hugo_extended_0.91.2_Linux-64bit.deb
RUN dpkg -i hugo_extended_0.91.2_Linux-64bit.deb

WORKDIR /workdir
