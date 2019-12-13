// Init variables
var cellw = 16;
var canvasw = 28 * cellw;
var pts = 0;
var data = new Array(28*28).fill(0);
var canvas, ctx, flag = false, dot_flag = false;

/*
    Inits the drawing canvas
*/
function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
    
    draw_grid();
    show_data();
}

/*
    Shows the digit data array as a matrix
*/
function show_data() {
    // Show digit data
    str = "";
    for (i = 0; i < data.length; i++) {
        if (i % 28 == 0 && i > 0) {
            str += "<br>";
        }

        if (data[i] == 0) {
            str += "<font color='#cccccc'>" + data[i] + "</font> ";
        }
        else {
            str += "<font color='green'>" + data[i] + "</font> ";
        }

    }
    document.getElementById("data").innerHTML = str;
}

/*
    Fills a cell with dark-green square
*/
function drawcell(x, y) {
    // Convert to current cell
    marginToCanvas = 10;
    x1 = x - x % cellw;
    y1 = y - y % cellw;
    // Draw square
    ctx.beginPath();
    ctx.fillStyle = "#1D6819";
    ctx.fillRect(x1, y1, cellw, cellw);
    ctx.closePath();
    
    // Put to data array
    i = x1/cellw + y1/cellw * 28;
    if (i >= 0 && i < 784) {
        if (data[i] == 0) {
            // New point
            data[i] = 1;
            pts++;
        }
    }
}

/*
    Draws one gridline in the canvas grid
*/
function gridline(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = "#bbbbbb";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.closePath();
}

/*
    Draws the canvas grid
*/
function draw_grid() {
    // Vertical grid lines
    for (c = 1; c <= 28; c++) {
        gridline(c * cellw, 0, c * cellw, canvasw);
    }
    // Horizontal grid lines
    for (r = 1; r <= 28; r++) {
        gridline(0, r * cellw, canvasw, r * cellw);
    }
}

/*
    Clears the canvas and data array
*/
function erase() {
    ctx.clearRect(0, 0, w, h);
    document.getElementById("canvasimg").style.display = "none";

    // Clear data div
    document.getElementById("data").innerHTML = "";
    data = new Array(28*28).fill(0);
    pts = 0;
    // Clear results
    document.getElementById("label").innerHTML = "&nbsp;";
    document.getElementById("prob").innerHTML = "";
    document.getElementById("probs").innerHTML = "";

    draw_grid();
    show_data();
}

/*
    Classifies the drawn digit
*/
function classify() {
    // Requires at least some drawn points
    if (pts <= 10) {
        return;
    }

    // Show digit data
    show_data();

    // Send classification request to backend
    doClassify();
}

/*
    Draw in the canvas
*/
function findxy(res, e) {
    // Click
    if (res == 'down') {
        x = e.clientX - canvas.offsetLeft;
        y = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            drawcell(x, y);
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    // Click and drag
    if (res == 'move') {
        if (flag) {
            x = e.clientX - canvas.offsetLeft;
            y = e.clientY - canvas.offsetTop;
            drawcell(x, y);
        }
    }
}


/*
    Sends classification request to backend
*/
function doClassify()
{
    // Convert data array to string
    p = "";
    for (i = 0; i < data.length; i++) {
        p += data[i];
    }

    // Make call to backend
    $(document).ready(function() {
        $.getJSON("classify","data="+p, function(res) {
            p = res.prob * 100;
            if (p >= 50) {
                // Print label
                document.getElementById("label").innerHTML = "<font color='blue'><b>" + res.label + "</font></b>";

                // Set color based on probability
                c = "red";
                if (p > 70) c = "orange";
                if (p > 90) c = "green";
                // Set probability
                document.getElementById("prob").innerHTML = "<font color='" + c + "'><b>" + p.toFixed(1) + "%</b></font>";
            }
            else {
                document.getElementById("label").innerHTML = "<font color='blue'>Don't know</font>";
                document.getElementById("prob").innerHTML = "";
            }

            // Probabilities for all digits
            str = "";
            for (i = 0; i < 10; i++) {
                p = res.probs[i] * 100;
                sp = "";
                if (p < 99.95) {
                    sp = "&nbsp;";
                }
                if (p < 9.95) {
                    sp = "&nbsp;&nbsp;";
                }
                str += "<b>" + i + "</b>: " + sp + "" + p.toFixed(1) + "%<br>"
            }
            document.getElementById("probs").innerHTML = str;

        });
    });
}