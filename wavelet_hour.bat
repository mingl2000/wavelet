echo off
set arg1=%1
python .\mingwave4_CN_clean.py %arg1% 256 1h True 20 True 128 0.1
REM python .\mingwave.py %arg1% 256 1h True 20 False 128
REM  中国平安  602318
REM python .\mingwave.py 002049.sz 256 1d True 20 True 128
REM python .\mingwave.py 601318.ss 256 1d True 20 True 128