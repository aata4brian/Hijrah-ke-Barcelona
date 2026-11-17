// Lightweight logic/DOM-contract tests; not a replacement for a real browser.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'Viewer_Hijrah_ke_Barcelona.html'), 'utf8');
class Element {
  constructor() { this.dataset = {}; this.children = []; this.events = {}; this.parts = {}; this.paused = true; this.duration = 7972.76; }
  set innerHTML(value) { this.children = [new Element(), new Element()]; }
  querySelector(selector) { return this.parts[selector] ||= new Element(); }
  addEventListener(name, fn) { this.events[name] = fn; }
  append(element) { this.children.push(element); }
  replaceChildren() { this.children = []; }
  setAttribute(name, value) { this[name] = value; }
  pause() { this.paused = true; }
}
const elements = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element()]));
const context = vm.createContext({console, setTimeout, clearTimeout, URL, TextDecoderStream,
  document: {getElementById: id => { assert.ok(elements[id], `Missing DOM id ${id}`); return elements[id]; },
    createElement: () => new Element(), addEventListener() {}}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
vm.runInContext(`
  if (rail.length !== 18) throw Error('Expected two nine-frame rails');
  const text = 'frame,label\\n' + Array.from({length:TOTAL},(_,i)=>i+','+'HADX'[i%4]).join('\\n');
  const parsed=loadLabels(text,false);
  savePrediction('pred',parsed);
  savePrediction('pred2',{labels:parsed.labels.map(x=>x==='H'?'A':x)});
  state.ready=true;
  for (let frame=0;frame<8;frame++) {
    showFrame(frame,'Synthetic test');
    if ($('prediction').dataset.label!==state.pred[frame]) throw Error('CSV1 readout stale');
    if ($('prediction2').dataset.label!==state.pred2[frame]) throw Error('CSV2 readout stale');
    for (const {offset,button,key} of rail) {
      const i=frame+offset;
      if (button.dataset.label!==(i<0?'none':state[key][i])) throw Error('Frame rail stale');
    }
  }
  if (state.differences!==Math.ceil(TOTAL/4)) throw Error('Difference count wrong');
  let rejected=false;try {loadLabels(text.replace('0,H','1,H'),false);}catch(e){rejected=true;}
  if (!rejected) throw Error('Invalid frame sequence accepted');
  state.frame=0; $('nextDifference').events.click();
  if (state.pending!==4) throw Error('Next difference should skip current disagreement');
  state.pending=null;state.frame=TOTAL-1;render();
  if (rail.filter(x=>x.offset>0).some(x=>!x.button.disabled)) throw Error('Out-of-bounds frame enabled');
`, context);
assert.match(html, /#prediction, #prediction2\s*\{\s*background:\s*var\(--label-bg/);
assert.match(html, /#rail \.tag, #rail2 \.tag\s*\{\s*background:\s*var\(--label-bg/);
assert.doesNotMatch(html, /background:\s*var\(--(?:csv1|csv2|frames1|frames2)\)/);
assert.match(html, /--H: #368ef5; --A: #ff3744; --D: #ffd43b; --X: #aeb4be/);
console.log('PASS: dual readouts, 18 frame cells, frame changes, H/A/D/X, CSV validation, disagreement navigation, bounds, dynamic CSS contract. Browser rendering/video playback not tested.');
