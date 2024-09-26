let mediaStream = null;
document.querySelector("#cameraSelect").addEventListener("change", (e) => {
  startCameraSelected(document.querySelector("#cameraSelect").value);
  openCamera();
});

document.querySelector("#btn-upload-images").addEventListener("click", (e) => {
  e.preventDefault();
  $('#modalProgressbar').modal('show');
  const http = new XMLHttpRequest();
  let formData = new FormData(document.querySelector('#form-images'));
  formData.append('name', document.querySelector('#name').value.trim());
  formData.append('ci', document.querySelector('#ci').value.trim());
  http.open("POST", "/save-faces");
  http.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let data = this.responseText;
      data = JSON.parse(data);
      $('#modalProgressbar').modal('hide');
      cleanForm();
      showMessage(data.message);
    }
  }
  http.send(formData);
});

$("#container-images").on("click", ".btn-remove", (e) => {
  e.target.parentElement.remove();
});

function init() {
  getSelectDevices();
}

function cleanForm() {
  document.querySelector('#name').value = '';
  document.querySelector('#ci').value = '';
  document.querySelector('#container-images').innerHTML = '';
  visibiliBtnForm();
}

function visibiliBtnForm() {
  if (document.querySelector("#container-images").childElementCount == 0)
    document.querySelector("#btn-upload-images").classList.add("d-none");
  else document.querySelector("#btn-upload-images").classList.remove("d-none");
}

function openCamera() {
  const video = document.getElementById("video");

  document.getElementById("camera-close").style.display = "block";
  document.getElementById("camera-button").style.display = "none";

  video.style.display = "block";
}

function startCameraSelected(deviceId) {
  let captureButton = document.getElementById("capture-button");

  navigator.mediaDevices
    .getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined },
      audio: false,
    })
    .then(function (stream) {
      mediaStream = stream;
      video.srcObject = stream;
      captureButton.style.display = "block";
    })
    .catch(function (error) {
      console.error("Error al acceder a la cámara: ", error);
      alert("Error al acceder a la cámara: ", error);
    });
}

async function getSelectDevices() {
  try {
    const cameraSelect = document.getElementById("cameraSelect");
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(
      (device) => device.kind === "videoinput"
    );

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    /* if (videoDevices.length > 0) {
      document.querySelector("#cameraSelect").value = videoDevices[0].deviceId;
      startCameraSelected(videoDevices[0].deviceId);
    } */
  } catch (err) {
    console.error("Error accessing devices: ", err);
  }
}

function capturePhoto() {
  const video = document.getElementById("video");

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convertir la imagen a formato base64
  const imageDataURL = canvas.toDataURL("image/jpeg");

  let html = `
    <div class="container-list-image col-12 col-md-5 col-lg-4 m-2">
        <img src="${imageDataURL}"/>
        <input type="hidden" name="photo[]" value="${imageDataURL}">
        <button class="btn btn-block btn-danger btn-remove">Limpiar</button>
    </div>`;
  $("#container-images").append(html);

  visibiliBtnForm();

  /* if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  } */

  //   closeCamera();
}

document.getElementById("camera-button").addEventListener("click", openCamera);
document
  .getElementById("capture-button")
  .addEventListener("click", capturePhoto);
document.getElementById("camera-close").addEventListener("click", closeCamera);

function closeCamera() {
  document.getElementById("camera-button").style.display = "block";
  document.getElementById("camera-close").style.display = "none";
  // document.getElementById("video").style.display = "none";
  document.getElementById("capture-button").style.display = "none";
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
}

function showMessage(message) {
  Swal.fire(message);
}

init();
