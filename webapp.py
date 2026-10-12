from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    if not message:
        prediction = "Please enter a message."
    else:
        prediction = "Message received: " + message

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    
    app.run(debug=True)