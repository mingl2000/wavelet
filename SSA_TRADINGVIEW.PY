// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © loxx

//@version=5
indicator("SSA of Price [Loxx]",
     shorttitle = "SSP [Loxx]", 
     overlay = true, 
     max_lines_count = 500)
     
import loxx/loxxexpandedsourcetypes/4

greencolor = #2DD204
redcolor = #D2042D 

int Maxncomp = 25
int MaxLag = 200
int MaxArrayLength = 1000

// Calculation of the function Sn, needed to calculate the eigenvalues
// Negative determinants are counted there 
gaussSn(matrix<float> A, float l, int n)=>
    array<float> w = array.new<float>(n, 0)
    matrix<float> B = matrix.copy(A)
    int count = 0
    int cp = 0
    float c = 0.
    float s1 = 0.
    float s2 = 0.
    for i = 0 to n - 1 
        matrix.set(B, i, i, matrix.get(B, i, i) - l)
    for k = 0 to n - 2 
        for i = k + 1 to n - 1 
            if matrix.get(B, k, k) == 0
                for i1 = 0 to n - 1
                    array.set(w, i1, matrix.get(B, i1, k)) 
                    matrix.set(B, i1, k, matrix.get(B, i1, k + 1))
                    matrix.set(B, i1, k + 1, array.get(w, i1))
                cp := cp + 1
            c := matrix.get(B, i, k) / matrix.get(B, k, k) 
            for j = 0 to n - 1 
                matrix.set(B, i, j, matrix.get(B, i, j) - matrix.get(B, k, j) * c)
    count := 0
    s1 := 1
    for i = 0 to n - 1 
        s2 := matrix.get(B, i, i) 
        if s2 < 0
            count := count + 1
    count

// Calculation of eigenvalues by the bisection method}
// The good thing is that as many eigenvalues are needed, so many will count,
// saves a lot of resources
gaussbisectionl(matrix<float> A, int k, int n)=>
    float e1 = 0.
    float maxnorm = 0.
    float cn = 0.
    float a1 = 0.
    float b1 = 0.
    float c = 0.
    for i = 0 to n - 1 
        cn := 0
        for j = 0 to n - 1 
            cn := cn + matrix.get(A, i, i) 
        if maxnorm < cn 
            maxnorm := cn
    a1 := 0
    b1 := 10 * maxnorm
    e1 := 1.0 * maxnorm / 10000000
    while math.abs(b1 - a1) > e1
        c := 1.0 * (a1 + b1) / 2
        if gaussSn(A, c, n) < k
            a1 := c
        else
            b1 := c
    float out = (a1 + b1) / 2.0
    out

// Calculates eigenvectors for already computed eigenvalues 
svector(matrix<float> A, float l, int n, array<float> V)=>
    int cp = 0
    matrix<float> B = matrix.copy(A)
    float c = 0
    array<float> w = array.new<float>(n, 0)
    for i = 0 to n - 1 
        matrix.set(B, i, i, matrix.get(B, i, i) - l)
    for k = 0 to n - 2 
        for i = k + 1 to n - 1
            if matrix.get(B, k, k) == 0
                for i1 = 0 to n - 1 
                    array.set(w, i1, matrix.get(B, i1, k))
                    matrix.set(B, i1, k, matrix.get(B, i1, k + 1))
                    matrix.set(B, i1, k + 1, array.get(w, i1))
                cp += 1
            
            c := 1.0 * matrix.get(B, i, k) / matrix.get(B, k, k) 
            for j = 0 to n - 1 
                matrix.set(B, i, j, matrix.get(B, i, j) - matrix.get(B, k, j) * c)
    array.set(V, n - 1, 1)
    c := 1
    for i = n - 2 to 0 
        array.set(V, i, 0) 
        for j = i to n - 1 
            array.set(V, i, array.get(V, i) - matrix.get(B, i, j) * array.get(V, j))
        array.set(V, i, array.get(V, i) / matrix.get(B, i, i))
        c += math.pow(array.get(V, i), 2)
    for i = 0 to n - 1 
        array.set(V, i, array.get(V, i) / math.sqrt(c))

// Fast Singular SSA - "Caterpillar" method
// X-vector of the original series
// n-length
// l-lag length
// s-number of eigencomponents
// (there the original series is divided into components, and then restored, here you set how many components you need)
// Y - the restored row (smoothed by the caterpillar) 
fastsingular(array<float> X, int n1, int l1, int s1)=>
    int n = math.min(MaxArrayLength, n1)
    int l = math.min(MaxLag, l1)
    int s = math.min(Maxncomp, s1)
    
    matrix<float> A = matrix.new<float>(l, l, 0.)
    matrix<float> B = matrix.new<float>(n, l, 0.)
    matrix<float> Bn = matrix.new<float>(l, n, 0.)
    matrix<float> V = matrix.new<float>(l, n, 0.)
    matrix<float> Yn = matrix.new<float>(l, n, 0.)
    
    var array<float> vtarr = array.new<float>(l, 0.)
    array<float> ls = array.new<float>(MaxLag, 0)
    array<float> Vtemp = array.new<float>(MaxLag, 0)
    array<float> Y = array.new<float>(n, 0)
    
    int k = n - l + 1
    
    // We form matrix A in the method that I downloaded from the site of the creators of this matrix S 
    for i = 0 to l - 1  
        for j = 0 to l - 1 
            matrix.set(A, i, j, 0)
            for m = 0 to k - 1 
                matrix.set(A, i, j, matrix.get(A, i, j) + array.get(X, i + m) * array.get(X, m + j))
                matrix.set(B, m, j, array.get(X, m + j))

    //Find the eigenvalues and vectors of the matrix A
    for i = 0 to s - 1  
        array.set(ls, i, gaussbisectionl(A, l - i, l))
        svector(A, array.get(ls, i), l, Vtemp)  
        for j = 0 to l - 1 
			matrix.set(V, i, j, array.get(Vtemp, j))

    // The restored matrix is formed
    for i1 = 0 to s - 1
        for i = 0 to k - 1  
            matrix.set(Yn, i1, i, 0) 
            for j = 0 to l - 1  
				matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(B, i, j) * matrix.get(V, i1, j))

        for i = 0 to l - 1 
            for j = 0 to k - 1 
				matrix.set(Bn, i, j, matrix.get(V, i1, i) * matrix.get(Yn, i1, j))
				
        //Diagonal averaging (series recovery)
        kb = k
        lb = l
        for i = 0 to n - 1  
            matrix.set(Yn, i1, i, 0)
            if i < lb - 1 
                for j = 0 to i
                    if l <= k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, j, i - j))
                    if l > k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, i - j, j))
                
                matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) / (1.0 * (i + 1)))
            if (lb - 1 <= i) and (i < kb - 1)
                for j = 0 to lb - 1  
                    if l <= k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, j, i - j))
                    if l > k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, i - j, j))
                matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) / (1.0 * lb))
            if kb - 1 <= i 
                for j = i - kb + 1 to n - kb  
                    if l <= k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, j, i - j))
                    if l > k
						matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) + matrix.get(Bn, i - j, j))
    
                matrix.set(Yn, i1, i, matrix.get(Yn, i1, i) / (1.0 * (n - i))) 

    // Here, if not summarized, then there will be separate decomposition components
    // process by own functions 
    for i = 0 to n - 1
        array.set(Y, i, 0) 
        for i1 = 0 to s - 1  
            array.set(Y, i, array.get(Y, i) + matrix.get(Yn, i1, i))
    Y

smthtype = input.string("Kaufman", "Heiken-Ashi Better Smoothing", options = ["AMA", "T3", "Kaufman"], group=  "Source Settings")
srcoption = input.string("Close", "Source", group= "Source Settings", 
     options = 
     ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted", "Average", "Average Median Body", "Trend Biased", "Trend Biased (Extreme)", 
     "HA Close", "HA Open", "HA High", "HA Low", "HA Median", "HA Typical", "HA Weighted", "HA Average", "HA Average Median Body", "HA Trend Biased", "HA Trend Biased (Extreme)",
     "HAB Close", "HAB Open", "HAB High", "HAB Low", "HAB Median", "HAB Typical", "HAB Weighted", "HAB Average", "HAB Average Median Body", "HAB Trend Biased", "HAB Trend Biased (Extreme)"])

lag = input.int(10, "Lag", group = "Bands Settings")
ncomp = input.int(2, "Number of Computations", group = "Bands Settings")
ssapernorm = input.int(20, "SSA Period Normalization", group = "Bands Settings")
numbars = input.int(300, "Number of Bars", group = "Bands Settings")

colorbars = input.bool(false, "Mute bars?", group = "UI Options")

kfl=input.float(0.666, title="* Kaufman's Adaptive MA (KAMA) Only - Fast End", group = "Moving Average Inputs")
ksl=input.float(0.0645, title="* Kaufman's Adaptive MA (KAMA) Only - Slow End", group = "Moving Average Inputs")
amafl = input.int(2, title="* Adaptive Moving Average (AMA) Only - Fast", group = "Moving Average Inputs")
amasl = input.int(30, title="* Adaptive Moving Average (AMA) Only - Slow", group = "Moving Average Inputs")

haclose = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close)
haopen = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, open)
hahigh = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, high)
halow = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, low)
hamedian = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, hl2)
hatypical = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, hlc3)
haweighted = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, hlcc4)
haaverage = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, ohlc4)

float src = switch srcoption
	"Close" => loxxexpandedsourcetypes.rclose()
	"Open" => loxxexpandedsourcetypes.ropen()
	"High" => loxxexpandedsourcetypes.rhigh()
	"Low" => loxxexpandedsourcetypes.rlow()
	"Median" => loxxexpandedsourcetypes.rmedian()
	"Typical" => loxxexpandedsourcetypes.rtypical()
	"Weighted" => loxxexpandedsourcetypes.rweighted()
	"Average" => loxxexpandedsourcetypes.raverage()
    "Average Median Body" => loxxexpandedsourcetypes.ravemedbody()
	"Trend Biased" => loxxexpandedsourcetypes.rtrendb()
	"Trend Biased (Extreme)" => loxxexpandedsourcetypes.rtrendbext()
	"HA Close" => loxxexpandedsourcetypes.haclose(haclose)
	"HA Open" => loxxexpandedsourcetypes.haopen(haopen)
	"HA High" => loxxexpandedsourcetypes.hahigh(hahigh)
	"HA Low" => loxxexpandedsourcetypes.halow(halow)
	"HA Median" => loxxexpandedsourcetypes.hamedian(hamedian)
	"HA Typical" => loxxexpandedsourcetypes.hatypical(hatypical)
	"HA Weighted" => loxxexpandedsourcetypes.haweighted(haweighted)
	"HA Average" => loxxexpandedsourcetypes.haaverage(haaverage)
    "HA Average Median Body" => loxxexpandedsourcetypes.haavemedbody(haclose, haopen)
	"HA Trend Biased" => loxxexpandedsourcetypes.hatrendb(haclose, haopen, hahigh, halow)
	"HA Trend Biased (Extreme)" => loxxexpandedsourcetypes.hatrendbext(haclose, haopen, hahigh, halow)
	"HAB Close" => loxxexpandedsourcetypes.habclose(smthtype, amafl, amasl, kfl, ksl)
	"HAB Open" => loxxexpandedsourcetypes.habopen(smthtype, amafl, amasl, kfl, ksl)
	"HAB High" => loxxexpandedsourcetypes.habhigh(smthtype, amafl, amasl, kfl, ksl)
	"HAB Low" => loxxexpandedsourcetypes.hablow(smthtype, amafl, amasl, kfl, ksl)
	"HAB Median" => loxxexpandedsourcetypes.habmedian(smthtype, amafl, amasl, kfl, ksl)
	"HAB Typical" => loxxexpandedsourcetypes.habtypical(smthtype, amafl, amasl, kfl, ksl)
	"HAB Weighted" => loxxexpandedsourcetypes.habweighted(smthtype, amafl, amasl, kfl, ksl)
	"HAB Average" => loxxexpandedsourcetypes.habaverage(smthtype, amafl, amasl, kfl, ksl)
    "HAB Average Median Body" => loxxexpandedsourcetypes.habavemedbody(smthtype, amafl, amasl, kfl, ksl)
	"HAB Trend Biased" => loxxexpandedsourcetypes.habtrendb(smthtype, amafl, amasl, kfl, ksl)
	"HAB Trend Biased (Extreme)" => loxxexpandedsourcetypes.habtrendbext(smthtype, amafl, amasl, kfl, ksl)
	=> haclose
	
var pvlines = array.new_line(0)

if barstate.isfirst
    for i = 0 to 500 - 1
        array.push(pvlines, line.new(na, na, na, na))
        
if barstate.islast
    arr = array.new_float(numbars + 1, 0)
    for i = 0 to numbars - 1
        array.set(arr, i, nz(src[i]))
        
    pv = fastsingular(arr, numbars, lag, ncomp)
    sizer = array.size(pv)

    skipperpv = array.size(pv) >= 2000 ? 8 : array.size(pv) >= 1000 ? 4 : array.size(pv) >= 500 ? 2 : 1
    int i = 0
    int j = 0

    while i < sizer - 1 - skipperpv
        if j > array.size(pvlines) - 1
            break
            
        colorout = i < array.size(pv) - 2 ? array.get(pv, i) > array.get(pv, i + skipperpv) ? greencolor : redcolor : na

        pvline = array.get(pvlines, j)
        line.set_xy1(pvline, bar_index - i - skipperpv, array.get(pv, i + skipperpv))
        line.set_xy2(pvline, bar_index - i, array.get(pv, i))
        line.set_color(pvline, colorout)
        line.set_style(pvline, line.style_solid)
        line.set_width(pvline, 5)

        i += skipperpv
        j += 1
        
barcolor(colorbars ? color.black : na)

