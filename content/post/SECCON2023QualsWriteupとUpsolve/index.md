---
title: "SECCON CTF 2023 WriteUpとUpSolve"
date: 2023-09-16 18:00:00 +0900
math: true
categories:
    - CTF
---
## はじめに
今回、チーム「SUSH1st」のメンバーとして9/16から9/17まで開催されたSECCON CTF 2023 Qualsに参加しました。

## 開催中に解いた問題
### [Reversing] jumpout
渡されるファイルはELF形式の実行ファイル。


### [Reversing] Sickle


## 開催中に解けなかった問題
### [Pwn] ROP-2.35
タイトルや問題文的にBOFを起こしてROP gadgetをいじくる感じだろうと予想したが
問題文にもある通りROP gadgetが少ない。特に、関数の第一引数となるレジスタに値を入れることができる`pop rdi; ret`がないのは致命的。与えられたプログラム中にはgetsとsystemのGOTがあるのでこれをうまいこと使うのだろうか。使えたとしても、第一引数を自由にいじれないのをどうするべきか。

### [Misc] readme 2023
毎年恒例になっているらしいreadme問題。今回はmmapにflagが書き込まれている状態になっているらしい。
だが、どうやってmmapの中身を見ればいいのかわからない。

### [Misc] Tokyo Payload
諸事情によりここ1週間で何度も触ってきたsolidityの問題。
contractが何かなどは分かるようにはなったが、assemblyや無名関数など分からない文法が大量にあった。

### [sandbox] crabox
動いているサーバーのプログラム`server.py`によると、標準入力から入力したRustのプログラムをmain関数に書き込んでコンパイルをしてくれるらしい。実行はしてくれない。
問題文にもある通りCompile-Time sandbox EscapeをRustで行えばよいのだが、FLAGがコメント中に入ること、stdoutとstderrがどっちも/dev/nullに入るため、このままだとターミナル上には表示されないことが問題。

### [Reversing] Perfect-blu
.iso形式のファイルであるが、中身はBDMVというブルーレイの保存形式の一つらしい。
こっからどうやって映像を復元するのか。