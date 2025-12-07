let chart = null;

async function predict() {
    const features = [
        parseFloat(document.getElementById("sl").value),
        parseFloat(document.getElementById("sw").value),
        parseFloat(document.getElementById("pl").value),
        parseFloat(document.getElementById("pw").value)
    ];

    if (features.some(isNaN)) {
        alert("Please fill all fields with valid numbers");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
    });

    const data = await response.json();

    if (data.detail) {
        document.getElementById("result").innerHTML = "❌ Error: " + data.detail;
        return;
    }

    // Show prediction text
    document.getElementById("result").innerHTML =
        `🌼 Predicted Class: <b>${data.prediction.toUpperCase()}</b>`;

    // Probability Chart
    const labels = ["setosa", "versicolor", "virginica"];
    const probabilities = data.probabilties;

    if (chart !== null) chart.destroy();

    const ctx = document.getElementById("probChart").getContext("2d");

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Prediction Probabilities",
                data: probabilities,
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, max: 1 }
            }
        }
    });
}
