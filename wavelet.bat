echo off
echo off
if [%1]==[]  goto :blank
if [%2]==[]  goto :blank
set arg1=%1
set arg2=%2

python .\mingwave4.py %arg1% 2048 %arg2% True 20 True 1280 0.1
goto :done
:blank
echo wavelet.bat QQQ 5m
echo wavelet.bat QQQ 1h
echo wavelet.bat QQQ 1d


echo python .\mingwave3.py QQQ 256 1d True 20 True 128
echo python .\mingwave3.py SPX 256 1d True 20 True 128

echo python .\mingwave3.py 399001.sz 256 1d True 20 True 128
echo python .\mingwave3.py 000001.ss 256 1d True 20 True 128

echo python .\mingwave3.py 002049.sz 256 1d True 20 True 128
echo python .\mingwave3.py 601318.ss 256 1d True 20 True 128
:done
echo done
