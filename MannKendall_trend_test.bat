if [%1]==[] goto :blank

set arg1=%1
python .\MannKendall_trend_test.py %arg1% 500 1d 13 13