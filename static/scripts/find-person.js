
const socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('person_recognized', function (data) {
  console.log(data);
  if(data.frame_find_person && (document.querySelector('#camera-close').style.display == 'block' || document.querySelector('#camera-close-2').style.display == 'block')) {
    if($('#cameraSelect :selected').val()  == data.camera)
      document.querySelector('#video').src = 'data:image/jpeg;base64,' + data.frame_find_person;
    else
      document.querySelector('#video-2').src = 'data:image/jpeg;base64,' + data.frame_find_person;
  }
});

$('#list-persons').on('click', '.check-list', e => {
  // startFindPerson($('#cameraSelect :selected').val());
});

document.querySelector('#cameraSelect').addEventListener('change', e => {
  verifySelectedCamera(1);
});
document.querySelector('#cameraSelect-2').addEventListener('change', e => {
  verifySelectedCamera(2);
});

document.getElementById("camera-button").addEventListener("click", (e) => {
  startFindPerson($('#cameraSelect :selected').val(), 1);
});

document.getElementById("camera-close").addEventListener("click", (e) => {
  stopFindPerson($('#cameraSelect :selected').val(), 1);
  closeCamera(1);
});

document.getElementById("camera-button-2").addEventListener("click", (e) => {
  startFindPerson($('#cameraSelect-2 :selected').val(), 2);
});

document.getElementById("camera-close-2").addEventListener("click", (e) => {
  stopFindPerson($('#cameraSelect-2 :selected').val(), 2);
  closeCamera(2);
});

function init() {
  getListPersons();
}

function verifySelectedCamera(index) {
  if($('#cameraSelect :selected').val() != '') {
    if($('#cameraSelect :selected').val() == $('#cameraSelect-2 :selected').val()) {
      let aux = index == 2 ? '-2' : '';
      document.querySelector('#cameraSelect'+aux).value = '';
      alert('Indice de cámara repetida.')
      return;
    }
  }
}

function startFindPerson(camera, index) {
  let data = getCheckSelected();
  if(!data.name) {
    alert("Seleccione una persona")
    return;
  }
  if(data.name && camera !== '') {
    openCamera(index);
    let formData = new FormData();
    formData.append('file', data.file);
    formData.append('name', data.name);
    formData.append('ci', data.ci);
    formData.append('camera', camera);
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

function stopFindPerson(camera, index) {
  let formData = new FormData();
  formData.append('camera', camera);
  const http = new XMLHttpRequest();
  http.open("POST", "/stop-find-person");
  http.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let data = this.responseText;
      console.log(data);
      let aux = index == 2 ? '-2' : '';
      document.getElementById("video"+aux).src = "static/images/camera.png";
    }
  }
  http.send(formData);
}

function getCheckSelected() {
  let checks = document.getElementsByClassName('check-list');
  let rspta = {
    name : '',
    ci : '',
    file : '',
  }
  // if(!$('#cameraSelect :selected').val()) return rspta;
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


function openCamera(index) {
  let aux = index == 2 ? '-2' : '';
  document.getElementById("camera-close"+aux).style.display = "block";
  document.getElementById("camera-button"+aux).style.display = "none";
}

function closeCamera(index) {
  let aux = index == 2 ? '-2' : '';
  document.getElementById("video"+aux).src = "static/images/camera.png";
  document.getElementById("camera-button"+aux).style.display = "block";
  document.getElementById("camera-close"+aux).style.display = "none";
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

