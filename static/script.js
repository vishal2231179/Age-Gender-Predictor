let video = document.getElementById("webcam");
let image = document.getElementById("uploadedImage");
let uploadInput = document.getElementById("uploadInput");
let resultBox = document.getElementById("result");
let stream;

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            video.style.display = "block";
            image.style.display = "none";
        })
        .catch(err => console.error("Error accessing camera:", err));
}

uploadInput.onchange = function (event) {
    let file = event.target.files[0];
    if (!file) {
        console.error("No file selected");
        return;
    }
    let reader = new FileReader();
    reader.onload = function () {
        image.src = reader.result;
        image.style.display = "block";
        video.style.display = "none";
    };
    reader.readAsDataURL(file);
};

function detectAgeGender() {
    let formData = new FormData();

    if (image.style.display === "block") {
        if (!uploadInput.files[0]) {
            alert("No image file found");
            return;
        }
        formData.append("file", uploadInput.files[0]);
        sendRequest(formData);
    } else if (video.style.display === "block") {
        // Capture a frame from the video
        let canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        let ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Convert the captured frame to a blob and send it
        canvas.toBlob(blob => {
            if (!blob) {
                console.error("Canvas conversion failed");
                alert("Failed to capture image from camera.");
                return;
            }
            formData.append("file", blob, "capture.jpg");
            sendRequest(formData);
        }, "image/jpeg");
    } else {
        alert("Please upload an image or start the camera!");
    }
}

function sendRequest(formData) {
    fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data && data.gender && data.age) {
            resultBox.innerHTML = `Detected: ${data.gender}, ${data.age} years old`;
        } else {
            resultBox.innerHTML = "Face detection failed. No results received.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Failed to detect age/gender. Check console for details.");
    });
}
