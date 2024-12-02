document.getElementById("predictionForm").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = ""; // Clear previous result

    // Collect form data
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    // Validate inputs
    for (const key in data) {
        if (data[key] === "" || parseFloat(data[key]) < 0) {
            resultDiv.innerHTML = `<p style="color: red;">Please provide valid positive values for all fields.</p>`;
            return;
        }
        data[key] = parseFloat(data[key]); // Convert values to numbers
    }

    try {
        // Send data to FastAPI backend
        const response = await fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error("Failed to get prediction. Please try again.");
        }

        const result = await response.json();
        resultDiv.innerHTML = `
            <p>Prediction: <strong>${result.prediction === 1 ? "Diabetes Detected" : "No Diabetes"}</strong></p>
            <p>Probability: <strong>${(result.probability * 100).toFixed(2)}%</strong></p>
        `;
    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
});
