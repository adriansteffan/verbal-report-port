# Automated Analysis of Verbal Reports
A port of a labmates experiment to work with this framework: https://github.com/tehillamo/AutoV-LLM

## Development
### Prerequisites
 - npm (>= 9.5.1), Node.js (>= 18.16.1), [ffmpeg](https://www.ffmpeg.org/download.html)

```bash
cd webserver
npm i
npm run dev
```
Available at `http://localhost:8080`

##  Production
### Prerequisites
 - [Docker](https://docs.docker.com/engine/install/)

```bash
cd webserver
docker compose -f docker-compose-prod.yaml up --build -d
```
Available at `http://localhost:8005`. Stop with `docker compose stop`.

---

## Create a Study using jsPsych with Automated Recording
The experiment is defined in [`/webserver/public/index.html`](/webserver/public/index.html) using [jsPsych](https://www.jspsych.org/). Wrap `jsPsychSpeechRecording` around your study trials:
```js
const trials = [
  { type: jsPsychSpeechRecording, start: true },
  // study trials here
  { type: jsPsychSpeechRecording, start: false }
];
jsPsych.run(trials);
```
The `jsPsychSpeechRecording` plugin is at [`/webserver/public/jspsych/dist/plugin-recording.js`](/webserver/public/jspsych/dist/plugin-recording.js). A template is at [`/webserver/public/index_template.html`](/webserver/public/index_template.html).

## Ressources
Recordings and trial data are saved to the [`ressources`](ressources) folder, in a subfolder per participant (identified by a random UUID).
