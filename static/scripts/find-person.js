
const socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('person_recognized', function (data) {
  console.log(data);
  if(data.frame_find_person) {
    document.querySelector('#video').src = 'data:image/jpeg;base64,' + data.frame_find_person;
  }
});

$('#list-persons').on('click', '.check-list', e => {
  startFindPerson();
});

document.getElementById("camera-button").addEventListener("click", (e) => {
  startFindPerson();
});

document.getElementById("camera-close").addEventListener("click", (e) => {
  stopFindPerson();
  closeCamera();
});

function init() {
  getListPersons();
}

function startFindPerson() {
  let data = getCheckSelected();
  if(!data.name) return;
  if(data.name && $('#cameraSelect :selected').val()) {
    openCamera();
    let formData = new FormData();
    formData.append('file', data.file);
    formData.append('name', data.name);
    formData.append('ci', data.ci);
    formData.append('camera', $('#cameraSelect :selected').val());
    const http = new XMLHttpRequest();
    http.open("POST", "/start-find-person");
    http.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
        let data = this.responseText;
        console.log(data);
      }
    }
    http.send(formData);
  } else {
    showMessage("Seleccione una cámara.");
  }
}

function stopFindPerson() {
  const http = new XMLHttpRequest();
  http.open("POST", "/stop-find-person");
  http.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let data = this.responseText;
      console.log(data);
      document.getElementById("video").src = "static/images/camera.png";
    }
  }
  http.send();
}

function getCheckSelected() {
  let checks = document.getElementsByClassName('check-list');
  let rspta = {
    name : '',
    ci : '',
    file : '',
  }
  for(let i = 0; i < checks.length; i++) {
    if(checks[i].checked) {
      rspta.name = checks[i].getAttribute('name');
      rspta.ci = checks[i].getAttribute('ci');
      rspta.file = checks[i].getAttribute('file');
    } else {
      checks[i].checked = false;
    }
  }
  return rspta;
}

function getListPersons() {
  const http = new XMLHttpRequest();
  http.open("POST", "/get-list-persons");
  http.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let data = this.responseText;
      data = JSON.parse(data);
      let aux = '';
      let html = '';
      data.data.forEach(element => {
        aux = splitText(element);
        html += `
          <div class="d-block">
            <input type="checkbox" file="${element}" name="${toCapitalize(aux.name)} ${toCapitalize(aux.surname)}" ci="${aux.ci}" class="check-list"> 
            ${toCapitalize(aux.name)} ${toCapitalize(aux.surname)}
          </div>`;
      });
      document.querySelector('#list-persons').innerHTML = html;
    }
  }
  http.send();
}


function openCamera() {
  document.getElementById("camera-close").style.display = "block";
  document.getElementById("camera-button").style.display = "none";
}

function closeCamera() {
  document.getElementById("video").src = "static/images/camera.png";
  document.getElementById("camera-button").style.display = "block";
  document.getElementById("camera-close").style.display = "none";
}

function splitText(nameFile) {
  let aux = nameFile.split('.jpg')[0];
  let ci = aux.split('__')[0];
  let auxName = aux.split('__')[1];
  let name = auxName.split('_')[0]
  let surname = auxName.split('_').length > 1 ? auxName.split('_')[1] : '';
  return {
    ci: ci,
    name: name,
    surname: surname,
  }
}

function toCapitalize(word) {
  word = word.toLowerCase();
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function showMessage(message) {
  Swal.fire(message);
}

init();

