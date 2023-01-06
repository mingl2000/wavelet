echo off
if [%1]==[] goto :blank

set arg1=%1
python .\ml.sr.py %arg1% 1m 0.01 12 10 4
goto :done
REM python .\ml.sr.py 000001.ss 1m 0.01 12 10 4
REM python .\ml.sr.py 399001.sz 1m 0.01 12 10 4
REM python .\ml.sr.py 002049.sz 1m 0.01 12 10 4
REM python .\ml.sr.py 601318.ss 1m 0.01 12 10 4
:blank
	echo python .\ml.sr.py 000001.ss 1m 0.01 12 10 4
	echo python .\ml.sr.py 399001.sz 1m 0.01 12 10 4
	echo python .\ml.sr.py 002049.sz 1m 0.01 12 10 4
	echo python .\ml.sr.py 601318.ss 1m 0.01 12 10 4

:done
	