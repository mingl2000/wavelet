echo off
echo off
if [%1]==[] goto :blank

set arg1=%1
python .\mingwave4_CN_clean.py %arg1% 256 1d True 20 True 128 0.1
goto :done
:blank
echo python .\mingwave3.py QQQ 256 1d True 20 True 128
echo python .\mingwave3.py SPX 256 1d True 20 True 128

echo python .\mingwave3.py 399001.sz 256 1d True 20 True 128
echo python .\mingwave3.py 000001.ss 256 1d True 20 True 128

echo python .\mingwave3.py 002049.sz 256 1d True 20 True 128
echo python .\mingwave3.py 601318.ss 256 1d True 20 True 128
:done
echo done
