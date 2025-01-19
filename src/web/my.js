function myfunc() {
    // const chart = LightweightCharts.createChart(document.body, { width: 400, height: 300 });
    const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
    const chart = LightweightCharts.createChart(document.getElementById('container'), chartOptions);

    const lineSeries = chart.addLineSeries();
    lineSeries.setData([
        { time: '2014-04-11', value: 80.01 },
        { time: '2019-04-12', value: 96.63 },
        { time: '2019-04-13', value: 76.64 },
        { time: '2019-04-14', value: 81.89 },
        { time: '2019-04-15', value: 74.43 },
        { time: '2019-04-16', value: 80.01 },
        { time: '2019-04-17', value: 96.63 },
        { time: '2019-04-18', value: 76.64 },
        { time: '2019-04-19', value: 81.89 },
        { time: '2019-04-20', value: 74.43 },
    ]);
}
function mycandle() {
    console.log("run mycanle");
    const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
    const chart = LightweightCharts.createChart(document.getElementById('container'), chartOptions);
    const areaSeries = chart.addAreaSeries({
        lineColor: '#2962FF', topColor: '#2962FF',
        bottomColor: 'rgba(41, 98, 255, 0.28)',
    });
    areaSeries.setData(convertData([
        { time: '20181222', value: 32.51 },
        { time: '20181223', value: 31.11 },
        { time: '20181224', value: 27.02 },
        { time: '20181225', value: 27.32 },
        { time: '20181226', value: 25.17 },
        { time: '20181227', value: 28.89 },
        { time: '20181228', value: 25.46 },
        { time: '20181229', value: 23.92 },
        { time: '20181230', value: 22.68 },
        { time: '20181231', value: 22.67 },
    ]));

    const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    candlestickSeries.setData(convertData([
        { time: '20181222', open: 75.16, high: 82.84, low: 36.16, close: 45.72 },
        { time: '20181223', open: 45.12, high: 53.90, low: 45.12, close: 48.09 },
        { time: '20181224', open: 60.71, high: 60.71, low: 53.39, close: 59.29 },
        { time: '20181225', open: 68.26, high: 68.26, low: 59.04, close: 60.50 },
        { time: '20181226', open: 67.71, high: 105.85, low: 66.67, close: 91.04 },
        { time: '20181227', open: 91.04, high: 121.40, low: 82.70, close: 111.40 },
        { time: '20181228', open: 111.51, high: 142.83, low: 103.34, close: 131.25 },
        { time: '20181229', open: 131.33, high: 151.17, low: 77.68, close: 96.43 },
        { time: '20181230', open: 106.33, high: 110.20, low: 90.39, close: 98.10 },
        { time: '20181231', open: 109.87, high: 114.69, low: 85.66, close: 111.26 },
    ]));

    chart.timeScale().fitContent();
}

function convertData(data) {
    return data.map(item => {
        const time = item.time;
        const year = time.slice(0, 4);
        const month = time.slice(4, 6);
        const day = time.slice(6, 8);
        return { ...item, time: `${year}-${month}-${day}` };
    });
}
function show_candle(data) {
    let areas_data = [];
    let candle_data = [];

    price = 1
    for (let item of data) {
        if (item.pe_cat === data[0].pe_cat) {
            areas_data.push({
                time: item.start_date,
                value: item.amount_yi/10000
            });
            tmp = price
            price = price * (item.pct_chg+1)
            candle_data.push({
                time: item.start_date,
                open: tmp,
                high: price,
                low: price,
                close: price
            });
        }
    }
    console.log(areas_data)
    console.log(candle_data)
    
    const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
    const chart = LightweightCharts.createChart(document.getElementById('container'), chartOptions);
    // const areaSeries = chart.addAreaSeries({
    //     lineColor: '#2962FF', topColor: '#2962FF',
    //     bottomColor: 'rgba(41, 98, 255, 0.28)',
    // });
    // areaSeries.setData(convertData(areas_data));

    const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    candlestickSeries.setData(convertData(candle_data));

    chart.timeScale().fitContent();
}
window.onload = function() {
    // myfunc();
    // mycandle();
};
