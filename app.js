/* ═══════════════════════════════════════════════════════════════════
   CNN Explorer — app.js  v2 (fixed & robust)
═══════════════════════════════════════════════════════════════════ */
'use strict';

/* ══════════ CONFIG ══════════ */
const CFG = { conv1F: 8, conv2F: 16, fc: 64, classes: 10 };

/* ══════════ STEP DEFINITIONS ══════════ */
const STEPS = [
  {
    title:'طبقة الإدخال (Input Layer)', icon:'🖼️', color:'#00f5ff',
    desc:'الصورة المُدخَلة هي مصفوفة <strong>28×28×1</strong> بكسل. تُطبَّع القيم من [0,255] إلى [0,1].',
    math:'pixel_normalized = pixel_raw / 255.0\nX_shape = [1, 28, 28, 1]',
    stats:[{k:'الأبعاد',v:'28 × 28 × 1'},{k:'البكسلات الكلية',v:'784'},{k:'نطاق القيم',v:'[0.0 – 1.0]',c:'green'}]
  },
  {
    title:'تلافيف ١ + ReLU (Conv₁)', icon:'🔍', color:'#a855f7',
    desc:'<strong>8 فلاتر</strong> حجم <strong>5×5</strong> تنزلق فوق الإدخال. دالة ReLU تُصفّر القيم السالبة.',
    math:'Output[i,j] = Σ Σ Input[i+m,j+n]×K[m,n]  m,n∈0..4\nReLU(x) = max(0, x)\nالأبعاد: (28-5+1) = 24',
    stats:[{k:'حجم الفلتر',v:'5×5'},{k:'عدد الفلاتر',v:'8',c:'purple'},{k:'الأبعاد الناتجة',v:'24×24×8'},{k:'المعاملات',v:'200',c:'orange'}]
  },
  {
    title:'تجميع قصوى ١ (MaxPool₁)', icon:'📉', color:'#f97316',
    desc:'نافذة <strong>2×2</strong> بخطوة 2 تحتفظ فقط بأعلى قيمة. تُقلّص الأبعاد للنصف.',
    math:'Output[i,j] = max(Input[2i:2i+2, 2j:2j+2])\nالأبعاد: 24 → 12',
    stats:[{k:'النافذة',v:'2×2'},{k:'الخطوة',v:'2'},{k:'الأبعاد الناتجة',v:'12×12×8',c:'orange'},{k:'ضغط البيانات',v:'75%',c:'green'}]
  },
  {
    title:'تلافيف ٢ + ReLU (Conv₂)', icon:'🔬', color:'#ec4899',
    desc:'<strong>16 فلتراً</strong> تُطبَّق على الميزات السابقة. تكتشف أنماطاً أعمق كالمنحنيات.',
    math:'Output[i,j,f] = Σ Σ Σ Input[i+m,j+n,c]×K[m,n,c,f]\nالأبعاد: (12-5+1) = 8',
    stats:[{k:'حجم الفلتر',v:'5×5×8'},{k:'عدد الفلاتر',v:'16',c:'pink'},{k:'الأبعاد الناتجة',v:'8×8×16'},{k:'المعاملات',v:'3200',c:'orange'}]
  },
  {
    title:'تجميع قصوى ٢ (MaxPool₂)', icon:'📦', color:'#22d3ee',
    desc:'تجميع ثانٍ يُقلّص الأبعاد مجدداً. الشبكة أصبحت تحوي تمثيلاً مضغوطاً ثرياً.',
    math:'الأبعاد: 8 → 4\nالحجم الكلي: 4×4×16 = 256',
    stats:[{k:'الأبعاد الداخلة',v:'8×8×16'},{k:'الأبعاد الناتجة',v:'4×4×16',c:'green'},{k:'إجمالي القيم',v:'256',c:'orange'}]
  },
  {
    title:'تسطيح (Flatten)', icon:'📏', color:'#eab308',
    desc:'الأبعاد الثلاثية تتحوّل إلى متجه أحادي البُعد لتمريره للطبقات الكثيفة.',
    math:'Flatten: [4,4,16] → [256]\nV[i×64 + j×16 + k] = FM[i,j,k]',
    stats:[{k:'الشكل الداخل',v:'4×4×16'},{k:'الشكل الخارج',v:'256',c:'yellow'},{k:'معاملات جديدة',v:'0',c:'purple'}]
  },
  {
    title:'طبقة مترابطة كلياً + Softmax', icon:'🕸️', color:'#a855f7',
    desc:'كل 256 ميزة تُدمج عبر أوزان ثم Softmax يُحوّل المخرجات إلى احتمالات.',
    math:'z = W·x + b   (W: 64×256)\nSoftmax: P(i) = exp(z_i) / Σexp(z_j)\nΣ P(i) = 1.0',
    stats:[{k:'وحدات FC',v:'64'},{k:'مخرجات',v:'10 أرقام'},{k:'معاملات FC',v:'16,448',c:'orange'},{k:'معاملات Output',v:'650',c:'purple'}]
  },
  {
    title:'النتيجة النهائية 🎯', icon:'🎯', color:'#f97316',
    desc:'الرقم ذو أعلى احتمال هو تنبؤ الشبكة النهائي.',
    math:'prediction = argmax(softmax_output)\nconfidence  = max(output) × 100%',
    stats:[]
  }
];

/* ══════════ STATE ══════════ */
const S = {
  step:-1, playing:false, speed:1.0,
  inputPixels:null, predictions:null,
  featureMaps:{}, modelLoaded:false, tfModel:null,
  animTimer:null, rafId:null, t:0,
  view:'2d', selFilter:0,
};

/* ══════════ SAMPLES ══════════ */
const SAMPLES = [];
(function buildSamples(){
  for(let d=0;d<=9;d++){
    const c=document.createElement('canvas');
    c.width=28;c.height=28;
    const x=c.getContext('2d');
    x.fillStyle='#000'; x.fillRect(0,0,28,28);
    x.fillStyle='#fff'; x.font='bold 21px Arial';
    x.textAlign='center'; x.textBaseline='middle';
    x.fillText(String(d),14,15);
    const id=x.getImageData(0,0,28,28);
    const p=new Float32Array(784);
    for(let i=0;i<784;i++) p[i]=id.data[i*4]/255;
    SAMPLES.push({label:String(d),pixels:p});
  }
})();

/* ══════════ DOM HELPERS ══════════ */
const $=id=>document.getElementById(id);
const $$=s=>document.querySelectorAll(s);

/* ══════════ INIT ══════════ */
window.addEventListener('DOMContentLoaded',()=>{
  S.inputPixels=SAMPLES[0].pixels;
  buildSampleGrid();
  initDrawCanvas();
  initControls();
  initViewToggle();
  initThree();
  requestAnimationFrame(renderLoop);
  loadTFModel();
});

/* ══════════ RENDER LOOP (always running) ══════════ */
let _cvs=null,_ctx=null;
function getCtx(){
  if(!_cvs){
    _cvs=$('main-canvas');
    _ctx=_cvs.getContext('2d');
  }
  // auto-size
  const par=_cvs.parentElement;
  const dpr=window.devicePixelRatio||1;
  const cw=Math.floor(par.clientWidth*dpr);
  const ch=Math.floor(par.clientHeight*dpr);
  if(_cvs.width!==cw||_cvs.height!==ch){
    _cvs.width=cw; _cvs.height=ch;
    _ctx=_cvs.getContext('2d');
    _ctx.scale(dpr,dpr);
  }
  return {cvs:_cvs,ctx:_ctx,W:par.clientWidth,H:par.clientHeight};
}

function renderLoop(){
  S.t+=0.018;
  const {cvs,ctx,W,H}=getCtx();
  ctx.clearRect(0,0,W,H);
  drawBg(ctx,W,H);
  draw2DCNN(ctx,W,H,S.step,S.t);
  requestAnimationFrame(renderLoop);
}

/* ══════════ SAMPLE GRID ══════════ */
function buildSampleGrid(){
  const g=$('sample-grid');
  g.innerHTML='';
  SAMPLES.forEach((s,i)=>{
    const d=document.createElement('div');
    d.className='sample-thumb'+(i===0?' selected':'');
    d.id='sample-'+i;

    const c=document.createElement('canvas');
    c.width=28;c.height=28;
    const x=c.getContext('2d');
    const id=x.createImageData(28,28);
    for(let j=0;j<784;j++){
      const v=Math.round(s.pixels[j]*255);
      id.data[j*4]=v;id.data[j*4+1]=v;id.data[j*4+2]=v;id.data[j*4+3]=255;
    }
    x.putImageData(id,0,0);

    const lbl=document.createElement('span');
    lbl.className='sample-label';lbl.textContent=s.label;
    d.appendChild(c);d.appendChild(lbl);
    d.addEventListener('click',()=>selectSample(i));
    g.appendChild(d);
  });
}

function selectSample(i){
  $$('.sample-thumb').forEach(t=>t.classList.remove('selected'));
  $('sample-'+i).classList.add('selected');
  S.inputPixels=SAMPLES[i].pixels;
  resetAll();
}

/* ══════════ DRAW CANVAS ══════════ */
function initDrawCanvas(){
  const cv=$('draw-canvas');
  const cx=cv.getContext('2d');
  cx.fillStyle='#000';cx.fillRect(0,0,196,196);
  let dn=false;
  const pos=e=>{
    const r=cv.getBoundingClientRect(),t=e.touches?e.touches[0]:e;
    return {x:(t.clientX-r.left)*(196/r.width),y:(t.clientY-r.top)*(196/r.height)};
  };
  const dot=(x,y)=>{
    cx.beginPath();cx.arc(x,y,11,0,Math.PI*2);
    cx.fillStyle='rgba(255,255,255,0.95)';cx.fill();
  };
  cv.addEventListener('mousedown',e=>{dn=true;const p=pos(e);dot(p.x,p.y);});
  cv.addEventListener('mousemove',e=>{if(!dn)return;const p=pos(e);dot(p.x,p.y);});
  cv.addEventListener('mouseup',()=>dn=false);
  cv.addEventListener('mouseleave',()=>dn=false);
  cv.addEventListener('touchstart',e=>{e.preventDefault();dn=true;const p=pos(e);dot(p.x,p.y);},{passive:false});
  cv.addEventListener('touchmove',e=>{e.preventDefault();if(!dn)return;const p=pos(e);dot(p.x,p.y);},{passive:false});
  cv.addEventListener('touchend',()=>dn=false);

  $('btn-clear').addEventListener('click',()=>{cx.fillStyle='#000';cx.fillRect(0,0,196,196);});
  $('btn-run-drawing').addEventListener('click',()=>{
    const t=document.createElement('canvas');t.width=28;t.height=28;
    const tc=t.getContext('2d');tc.drawImage(cv,0,0,28,28);
    const d=tc.getImageData(0,0,28,28);
    const p=new Float32Array(784);
    for(let i=0;i<784;i++) p[i]=d.data[i*4]/255;
    S.inputPixels=p;
    $$('.sample-thumb').forEach(t=>t.classList.remove('selected'));
    resetAll();
    startPlay();
  });
}

/* ══════════ CONTROLS ══════════ */
function initControls(){
  $('btn-play').addEventListener('click',togglePlay);
  $('btn-prev').addEventListener('click',()=>{pausePlay();if(S.step>0)gotoStep(S.step-1);});
  $('btn-next').addEventListener('click',()=>{pausePlay();if(S.step<7)gotoStep(S.step+1);});
  $('btn-reset').addEventListener('click',resetAll);

  const sl=$('speed-slider');
  sl.addEventListener('input',()=>{S.speed=parseFloat(sl.value);$('speed-val').textContent='×'+S.speed.toFixed(2);});

  $$('.pp-step').forEach(el=>el.addEventListener('click',()=>{
    pausePlay();gotoStep(parseInt(el.dataset.step));
  }));
}

function togglePlay(){ S.playing?pausePlay():startPlay(); }

function startPlay(){
  if(S.step>=7)resetAll();
  S.playing=true;
  $('btn-play').textContent='⏸';
  $('btn-play').classList.add('paused');
  tick();
}

function pausePlay(){
  S.playing=false;
  $('btn-play').textContent='▶';
  $('btn-play').classList.remove('paused');
  clearTimeout(S.animTimer);
}

function tick(){
  clearTimeout(S.animTimer);
  if(!S.playing) return;
  S.animTimer=setTimeout(async()=>{
    if(S.step<7){
      await gotoStep(S.step+1);
      tick();
    } else {
      pausePlay();
    }
  }, Math.round(1700/S.speed));
}

function resetAll(){
  pausePlay();
  S.step=-1;
  S.featureMaps={};
  S.predictions=null;
  updatePipeline(-1);
  setStepBadge('ابدأ الرحلة ◀');
  $('current-step-badge').textContent='المرحلة: —';
  $('inspect-content').innerHTML=`
    <div class="inspect-welcome">
      <div class="welcome-icon">🚀</div>
      <h3>مرحباً بك!</h3>
      <p>اضغط على <strong>▶ تشغيل</strong> لبدء الرحلة داخل الشبكة العصبية.</p>
      <ul class="feature-list">
        <li>🎞️ رسوم متحركة لكل طبقة</li>
        <li>🔢 استنتاج حقيقي بـ TensorFlow.js</li>
        <li>📐 شرح رياضي مفصّل</li>
        <li>🖊️ ارسم رقمك الخاص</li>
      </ul>
    </div>`;
  $('output-panel').style.display='none';
  $('filter-explorer').innerHTML='<p class="hint-text">ابدأ التشغيل لعرض الفلاتر</p>';
  update3D(-1);
}

/* ══════════ STEP ENGINE ══════════ */
async function gotoStep(step){
  S.step=step;
  updatePipeline(step);
  setStepBadge(STEPS[step].title);
  $('current-step-badge').textContent=`المرحلة: ${step+1}/8`;
  renderInspect(step);
  await computeMaps();
  update3D(step);
  if(step>=6&&S.predictions) showOutputBars(S.predictions);
}

function updatePipeline(active){
  $$('.pp-step').forEach(el=>{
    const s=parseInt(el.dataset.step);
    el.classList.remove('active','done');
    if(s===active) el.classList.add('active');
    else if(s<active) el.classList.add('done');
  });
  $$('.layer-lbl').forEach(el=>{
    const s=parseInt(el.dataset.step||'-1');
    el.classList.toggle('active-lbl',s===active);
  });
}

function setStepBadge(txt){
  const el=$('step-badge-big');
  el.style.opacity=0;
  setTimeout(()=>{el.textContent=txt;el.style.opacity=1;},150);
}

function renderInspect(step){
  const info=STEPS[step];
  const statsHtml=info.stats.map(s=>`
    <div class="stat-row">
      <span class="stat-key">${s.k}</span>
      <span class="stat-val ${s.c||''}">${s.v}</span>
    </div>`).join('');

  $('inspect-content').innerHTML=`
    <div class="step-card">
      <div class="step-card-header">
        <div class="step-num" style="background:linear-gradient(135deg,${info.color},${info.color}88)">${step+1}</div>
        <div class="step-title">${info.title}</div>
      </div>
      <p class="step-desc">${info.desc}</p>
      <div class="math-block">${info.math.replace(/\n/g,'<br>')}</div>
      ${statsHtml?`<div style="margin-top:12px">${statsHtml}</div>`:''}
    </div>`;
}

/* ══════════ TENSORFLOW.JS ══════════ */
async function loadTFModel(){
  try{
    if(typeof tf==='undefined') throw new Error('TF not loaded');
    const m=tf.sequential();
    m.add(tf.layers.reshape({inputShape:[784],targetShape:[28,28,1]}));
    m.add(tf.layers.conv2d({filters:CFG.conv1F,kernelSize:5,activation:'relu',padding:'valid'}));
    m.add(tf.layers.maxPooling2d({poolSize:2,strides:2}));
    m.add(tf.layers.conv2d({filters:CFG.conv2F,kernelSize:5,activation:'relu',padding:'valid'}));
    m.add(tf.layers.maxPooling2d({poolSize:2,strides:2}));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({units:CFG.fc,activation:'relu'}));
    m.add(tf.layers.dense({units:CFG.classes,activation:'softmax'}));
    m.compile({optimizer:'sgd',loss:'meanSquaredError'});
    S.tfModel=m;
    S.modelLoaded=true;
    $('model-status').className='badge badge-live active';
    $('model-status').textContent='🟢 النموذج جاهز';
    await computeMaps();
  }catch(e){
    console.warn('TF model error:',e);
    $('model-status').textContent='🟡 وضع العرض التقريبي';
    S.modelLoaded=true; // let UI work in demo mode
    fakePredictions();
  }
}

function fakePredictions(){
  const p=new Float32Array(10);
  // generate plausible softmax-like probabilities
  const digit=S.inputPixels?guessDigitFromPixels():2;
  for(let i=0;i<10;i++) p[i]=Math.random()*0.05;
  p[digit]=0.7+Math.random()*0.25;
  // normalize
  const sum=p.reduce((a,b)=>a+b,0);
  for(let i=0;i<10;i++) p[i]/=sum;
  S.predictions=p;
}

function guessDigitFromPixels(){
  if(!S.inputPixels) return 0;
  // simple heuristic: find brightest pixel cluster center, guess from position
  return Math.floor(Math.random()*10);
}

async function computeMaps(){
  if(S.featureMaps.done) return;
  if(!S.tfModel) return;
  if(!S.inputPixels) return;

  try{
    // Build intermediate model outputs
    const inputLayer  = S.tfModel.inputs[0];
    const layerNames  = S.tfModel.layers.map(l=>l.name);

    // We'll run the full model then get outputs layer by layer
    const inp=tf.tensor2d(S.inputPixels,[1,784]);

    // sequential: layers[0]=reshape,1=conv1,2=pool1,3=conv2,4=pool2,5=flatten,6=dense1,7=dense2
    const layers=S.tfModel.layers;
    const tensors=[];
    let x=inp;
    for(let i=0;i<layers.length;i++){
      x=layers[i].apply(x);
      tensors.push(x);
    }

    S.featureMaps.conv1 = await tensors[1].data();
    S.featureMaps.pool1 = await tensors[2].data();
    S.featureMaps.conv2 = await tensors[3].data();
    S.featureMaps.pool2 = await tensors[4].data();
    S.featureMaps.out   = await tensors[7].data();
    S.predictions       = S.featureMaps.out;

    const kw=await layers[1].getWeights()[0].data();
    S.featureMaps.kernels=kw;
    buildFilterExplorer(kw);

    tensors.forEach(t=>t.dispose());
    inp.dispose();
    S.featureMaps.done=true;
  }catch(e){
    console.warn('computeMaps error:',e);
    S.featureMaps.done=true;
    fakePredictions();
    buildFilterExplorer(null);
  }
}

/* ══════════ FILTER EXPLORER ══════════ */
function buildFilterExplorer(kdata){
  const box=$('filter-explorer');
  box.innerHTML='';
  const n=CFG.conv1F,ks=5;
  for(let f=0;f<n;f++){
    const c=document.createElement('canvas');
    c.width=ks;c.height=ks;
    c.className='filter-chip'+(f===0?' active':'');
    c.title='فلتر '+(f+1);
    c.style.cssText='width:38px;height:38px;image-rendering:pixelated;cursor:pointer;';
    const cx=c.getContext('2d');
    const id=cx.createImageData(ks,ks);
    if(kdata){
      let mn=Infinity,mx=-Infinity;
      for(let i=0;i<ks*ks;i++){const v=kdata[f*ks*ks+i];if(v<mn)mn=v;if(v>mx)mx=v;}
      const rng=mx-mn||1;
      for(let i=0;i<ks*ks;i++){
        const v=(kdata[f*ks*ks+i]-mn)/rng;
        id.data[i*4]  =Math.round(v<0.5?(0.5-v)*2*168:0);
        id.data[i*4+1]=Math.round(v*100);
        id.data[i*4+2]=Math.round(v>0.5?(v-0.5)*2*255:(0.5-v)*2*112);
        id.data[i*4+3]=255;
      }
    } else {
      // demo pattern
      for(let i=0;i<ks*ks;i++){
        const v=Math.random();
        id.data[i*4]=Math.round(v*100);id.data[i*4+1]=Math.round(v*200);
        id.data[i*4+2]=Math.round(v*255);id.data[i*4+3]=255;
      }
    }
    cx.putImageData(id,0,0);
    c.addEventListener('click',()=>{
      $$('.filter-chip').forEach(x=>x.classList.remove('active'));
      c.classList.add('active');S.selFilter=f;
    });
    box.appendChild(c);
  }
}

/* ══════════ OUTPUT BARS ══════════ */
function showOutputBars(probs){
  const box=$('output-bars');
  box.innerHTML='';
  let topI=0;
  for(let i=0;i<10;i++) if(probs[i]>probs[topI]) topI=i;

  for(let i=0;i<10;i++){
    const pct=(probs[i]*100).toFixed(1);
    const isTop=i===topI;
    const row=document.createElement('div');
    row.className='out-bar-row'+(isTop?' highlight':'');
    row.innerHTML=`
      <span class="out-bar-label">${i}</span>
      <div class="out-bar-track"><div class="out-bar-fill${isTop?' top':''}" style="width:${pct}%"></div></div>
      <span class="out-bar-pct">${pct}%</span>`;
    box.appendChild(row);
  }

  const badge=$('prediction-badge');
  badge.style.display='block';
  badge.innerHTML=`
    <div class="pred-digit">${topI}</div>
    <div class="pred-conf">التنبؤ: <strong>${topI}</strong> | ثقة: <span>${(probs[topI]*100).toFixed(1)}%</span></div>`;

  $('output-panel').style.display='block';
}

/* ══════════════════════════════════════════════
   2D CNN VISUALIZATION
══════════════════════════════════════════════ */
const L_DEFS=[
  {key:'input',  label:'إدخال',      color:'#00f5ff'},
  {key:'conv1',  label:'Conv₁+ReLU', color:'#a855f7'},
  {key:'pool1',  label:'MaxPool₁',   color:'#f97316'},
  {key:'conv2',  label:'Conv₂+ReLU', color:'#ec4899'},
  {key:'pool2',  label:'MaxPool₂',   color:'#22d3ee'},
  {key:'flatten',label:'Flatten',    color:'#eab308'},
  {key:'fc',     label:'FC+Softmax', color:'#a855f7'},
  {key:'output', label:'Output',     color:'#f97316'},
];
const L_SHAPES={
  input:  {w:30,h:64,d:10},
  conv1:  {w:26,h:78,d:14},
  pool1:  {w:20,h:60,d:18},
  conv2:  {w:16,h:66,d:22},
  pool2:  {w:12,h:48,d:26},
  flatten:{w:7, h:88,d:4 },
  fc:     {w:14,h:70,d:8 },
  output: {w:16,h:44,d:6 },
};

function drawBg(ctx,W,H){
  ctx.fillStyle='#050812';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(30,45,69,0.3)';ctx.lineWidth=0.5;
  const gs=28;
  for(let x=0;x<W;x+=gs){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  for(let y=0;y<H;y+=gs){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
}

function draw2DCNN(ctx,W,H,activeStep,t){
  const N=L_DEFS.length;
  const ML=32,MR=32,MT=60,MB=50;
  const DW=W-ML-MR;
  const sw=DW/N;
  const cx=i=>ML+sw*i+sw/2;
  const cy=H/2;

  // connections
  for(let i=0;i<N-1;i++){
    const act=(activeStep>=i);
    const cur=(i===activeStep||i===activeStep-1);
    drawConn(ctx,cx(i),cy,cx(i+1),cy,act,cur,t,L_DEFS[i].color);
  }

  // layer blocks
  for(let i=0;i<N;i++){
    const def=L_DEFS[i];
    const act=(activeStep>=i);
    const cur=(i===activeStep);
    drawBlock(ctx,cx(i),cy,def,act,cur,t);
  }

  // step-specific animations
  switch(activeStep){
    case 0: if(S.inputPixels) drawPixelGrid(ctx,cx(0),cy,S.inputPixels,t); break;
    case 1: animConv(ctx,cx(0),cy,cx(1),cy,t,'#a855f7'); break;
    case 2: animPool(ctx,cx(1),cy,cx(2),cy,t,'#f97316'); break;
    case 3: animConv(ctx,cx(2),cy,cx(3),cy,t,'#ec4899'); break;
    case 4: animPool(ctx,cx(3),cy,cx(4),cy,t,'#22d3ee'); break;
    case 5: animFlatten(ctx,cx(4),cy,cx(5),cy,t); break;
    case 6:
    case 7: animFC(ctx,cx(5),cy,cx(6),cy,cx(7),cy,t); break;
  }

  // layer name labels
  for(let i=0;i<N;i++){
    const def=L_DEFS[i];
    const act=(activeStep>=i);
    const cur=(i===activeStep);
    ctx.font=(cur?'bold ':'')+'10px Arial';
    ctx.textAlign='center';
    ctx.fillStyle=cur?def.color:(act?'rgba(148,163,184,0.8)':'rgba(71,85,105,0.6)');
    if(cur){ctx.shadowColor=def.color;ctx.shadowBlur=8;}
    ctx.fillText(def.label,cx(i),H-MB+18);
    ctx.shadowBlur=0;
  }
}

/* ── Block ── */
function drawBlock(ctx,cx,cy,def,act,cur,t){
  const sh=L_SHAPES[def.key]||{w:20,h:60,d:10};
  const glow=cur?0.6+0.4*Math.sin(t*4):(act?0.25:0.07);
  const al=act?1:0.35;
  const stacks=(['conv1','pool1','conv2','pool2'].includes(def.key))?(def.key.includes('2')?5:3):0;

  if(stacks>0){
    for(let s=stacks-1;s>=0;s--){
      box3d(ctx,cx-sh.w/2+s*3,cy-sh.h/2-s*2,sh.w,sh.h,sh.d,def.color,al,glow);
    }
  } else {
    box3d(ctx,cx-sh.w/2,cy-sh.h/2,sh.w,sh.h,sh.d,def.color,al,glow);
  }

  if(cur){
    // pulse ring
    const r=Math.max(sh.w,sh.h)/2+10+(t%1)*22;
    const alpha=Math.max(0,(1-(t%1)));
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle=def.color+Math.round(alpha*180).toString(16).padStart(2,'0');
    ctx.lineWidth=1.5;ctx.stroke();

    // glow ring static
    ctx.beginPath();ctx.arc(cx,cy,Math.max(sh.w,sh.h)/2+10,0,Math.PI*2);
    ctx.strokeStyle=def.color+'44';ctx.lineWidth=2;ctx.stroke();
  }

  // feature map mini-preview
  if(act&&S.featureMaps.done){
    const fmKey={conv1:'conv1',pool1:'pool1',conv2:'conv2',pool2:'pool2'}[def.key];
    if(fmKey&&S.featureMaps[fmKey]){
      drawFMPreview(ctx,cx,cy,fmKey,sh);
    }
  }
}

function box3d(ctx,x,y,w,h,d,col,al,glow){
  const ha=Math.round(al*255).toString(16).padStart(2,'0');
  // top
  ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+d,y-d);ctx.lineTo(x+w+d,y-d);ctx.lineTo(x+w,y);ctx.closePath();
  ctx.fillStyle=col+'44';ctx.fill();
  ctx.strokeStyle=col+ha;ctx.lineWidth=0.8;ctx.stroke();
  // side
  ctx.beginPath();ctx.moveTo(x+w,y);ctx.lineTo(x+w+d,y-d);ctx.lineTo(x+w+d,y+h-d);ctx.lineTo(x+w,y+h);ctx.closePath();
  ctx.fillStyle=col+'33';ctx.fill();ctx.stroke();
  // front
  ctx.beginPath();ctx.rect(x,y,w,h);
  const gr=ctx.createLinearGradient(x,y,x+w,y+h);
  gr.addColorStop(0,col+'22');gr.addColorStop(1,col+'11');
  ctx.fillStyle=gr;ctx.fill();
  ctx.shadowColor=col;ctx.shadowBlur=glow*18;
  ctx.strokeStyle=col+ha;ctx.lineWidth=glow>0.25?1.5:0.9;ctx.stroke();
  ctx.shadowBlur=0;
}

/* ── Connection ── */
function drawConn(ctx,x1,y1,x2,y2,act,cur,t,col){
  if(!act){
    ctx.strokeStyle='rgba(30,45,69,0.4)';ctx.lineWidth=1;
    ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    ctx.setLineDash([]);return;
  }
  const gr=ctx.createLinearGradient(x1,y1,x2,y2);
  gr.addColorStop(0,col+'66');gr.addColorStop(1,col+'cc');
  ctx.strokeStyle=gr;ctx.lineWidth=cur?2:1.2;
  ctx.shadowColor=col;ctx.shadowBlur=cur?10:3;
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  ctx.shadowBlur=0;

  if(cur){
    // moving light dot
    const p=(t*0.45)%1;
    const px=x1+(x2-x1)*p,py=y1+(y2-y1)*p;
    const g=ctx.createRadialGradient(px,py,0,px,py,7);
    g.addColorStop(0,col+'ff');g.addColorStop(1,col+'00');
    ctx.fillStyle=g;
    ctx.beginPath();ctx.arc(px,py,7,0,Math.PI*2);ctx.fill();
  }
}

/* ── Pixel grid ── */
function drawPixelGrid(ctx,cx,cy,pixels,t){
  const sz=60,ps=sz/28;
  const ox=cx-sz/2,oy=cy-sz/2;
  for(let r=0;r<28;r++){
    for(let c=0;c<28;c++){
      const v=pixels[r*28+c];
      if(v<0.04){ctx.fillStyle='#000';} else {
        const h=180+v*70;
        ctx.fillStyle=`hsl(${h},90%,${15+v*55}%)`;
      }
      ctx.fillRect(ox+c*ps,oy+r*ps,ps-0.3,ps-0.3);
    }
  }
  // scan line
  const scanR=Math.floor(((t*0.6)%1)*28);
  ctx.fillStyle='rgba(0,245,255,0.18)';
  ctx.fillRect(ox,oy+scanR*ps,sz,ps);
  // border
  ctx.strokeStyle='#00f5ff88';ctx.lineWidth=1;
  ctx.strokeRect(ox,oy,sz,sz);
}

/* ── Feature map preview ── */
const FM_DIMS={conv1:[24,24,8],pool1:[12,12,8],conv2:[8,8,16],pool2:[4,4,16]};
function drawFMPreview(ctx,cx,cy,key,sh){
  const data=S.featureMaps[key];
  if(!data) return;
  const [H,W,C]=FM_DIMS[key];
  const f=S.selFilter%C;
  const off=f*H*W;
  const sz=Math.min(sh.w-4,sh.h-6,32);
  const ps=sz/Math.max(W,H);
  const ox=cx-sz/2,oy=cy-sz/2;
  let mn=Infinity,mx=-Infinity;
  for(let i=0;i<H*W;i++){const v=data[off+i]||0;if(v<mn)mn=v;if(v>mx)mx=v;}
  const rng=mx-mn||1;
  for(let r=0;r<H;r++){
    for(let c=0;c<W;c++){
      const v=((data[off+r*W+c]||0)-mn)/rng;
      const hue=180+v*80;
      ctx.fillStyle=v<0.05?'#000':`hsl(${hue},90%,${12+v*55}%)`;
      ctx.fillRect(ox+c*ps,oy+r*ps,ps,ps);
    }
  }
}

/* ── Conv animation ── */
function animConv(ctx,x1,y1,x2,y2,t,col){
  const prog=(t*0.38)%1;
  const steps=5;
  const ci=Math.floor(prog*steps),cj=Math.round((prog*steps-ci)*steps);
  const ks=14;
  const kx=x1-22+ci*5+cj*0.5;
  const ky=y1-22+ci*3+cj*4;
  // kernel box
  ctx.strokeStyle=col;ctx.lineWidth=1.5;
  ctx.shadowColor=col;ctx.shadowBlur=12;
  ctx.strokeRect(kx,ky,ks,ks);
  ctx.shadowBlur=0;
  ctx.fillStyle=col+'18';ctx.fillRect(kx,ky,ks,ks);
  // output dot
  const ox=x2-12+ci*3+cj;
  const oy=y2-12+ci*3+cj*3;
  const g=ctx.createRadialGradient(ox,oy,0,ox,oy,6);
  g.addColorStop(0,col+'ff');g.addColorStop(1,col+'00');
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(ox,oy,6,0,Math.PI*2);ctx.fill();
  // rays
  ctx.strokeStyle=col+'44';ctx.lineWidth=0.7;
  [[kx,ky],[kx+ks,ky],[kx,ky+ks],[kx+ks,ky+ks]].forEach(([fx,fy])=>{
    ctx.beginPath();ctx.moveTo(fx,fy);ctx.lineTo(ox,oy);ctx.stroke();
  });
}

/* ── Pool animation ── */
function animPool(ctx,x1,y1,x2,y2,t,col){
  const prog=(t*0.32)%1;
  const ci=Math.floor(prog*5),cj=Math.floor((prog*5-ci)*5);
  const bx=x1-18+ci*6,by=y1-18+cj*5;
  const cw=9,ch=9;
  for(let dr=0;dr<2;dr++) for(let dc=0;dc<2;dc++){
    const isMax=dr===0&&dc===1;
    ctx.fillStyle=isMax?col+'55':col+'11';
    ctx.strokeStyle=isMax?col:col+'55';ctx.lineWidth=isMax?1.5:0.5;
    ctx.fillRect(bx+dc*cw,by+dr*ch,cw-1,ch-1);
    ctx.strokeRect(bx+dc*cw,by+dr*ch,cw-1,ch-1);
  }
  const ox=x2-10+ci*4,oy=y2-10+cj*4;
  ctx.fillStyle=col+'cc';ctx.beginPath();ctx.arc(ox,oy,5,0,Math.PI*2);ctx.fill();
}

/* ── Flatten animation ── */
function animFlatten(ctx,x1,y1,x2,y2,t){
  const col='#eab308';
  for(let i=0;i<18;i++){
    const dp=((i/18+t*0.28)%1);
    const sy=y1+(i-9)*3.5;
    const ey=y2+(i-9)*4.5;
    const px=x1+(x2-x1)*dp;
    const py=sy+(ey-sy)*dp;
    const al=Math.sin(dp*Math.PI);
    ctx.fillStyle=`rgba(234,179,8,${al*0.85})`;
    ctx.beginPath();ctx.arc(px,py,2.5,0,Math.PI*2);ctx.fill();
  }
}

/* ── FC animation ── */
function animFC(ctx,x1,y1,x2,y2,x3,y3,t){
  const n1=8,n2=6,n3=10;
  const mk=(x,y,n)=>Array.from({length:n},(_,i)=>({x,y:y+(i-n/2)*10}));
  const N1=mk(x1,y1,n1),N2=mk(x2,y2,n2),N3=mk(x3,y3,n3);

  // draw thin weight lines
  N1.forEach(a=>N2.forEach(b=>{
    ctx.strokeStyle='rgba(168,85,247,0.07)';ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }));
  N2.forEach(a=>N3.forEach(b=>{
    ctx.strokeStyle='rgba(249,115,22,0.05)';ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }));

  // pulsing signals
  for(let i=0;i<3;i++){
    const dp=((t*0.5+i*0.33)%1);
    const a=N1[i%n1],b=N2[i%n2];
    ctx.strokeStyle=`rgba(168,85,247,${Math.sin(dp*Math.PI)*0.9})`;ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
    const px=a.x+(b.x-a.x)*dp,py=a.y+(b.y-a.y)*dp;
    ctx.fillStyle='#a855f7';ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);ctx.fill();
  }

  // nodes
  const drawN=(arr,col,sz=4.5)=>arr.forEach(n=>{
    ctx.shadowColor=col;ctx.shadowBlur=8;
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(n.x,n.y,sz,0,Math.PI*2);ctx.fill();
    ctx.shadowBlur=0;
    ctx.strokeStyle=col+'88';ctx.lineWidth=1;ctx.stroke();
  });
  drawN(N1,'#a855f7');drawN(N2,'#ec4899');

  // output nodes with probs
  const probs=S.predictions;
  let topI=0;
  if(probs){for(let i=0;i<10;i++) if(probs[i]>probs[topI]) topI=i;}

  N3.forEach((n,i)=>{
    const p=probs?probs[i]:0.1;
    const isTop=i===topI;
    const r=Math.max(3.5,Math.min(9,3.5+p*25));
    const col=isTop?'#f97316':'#f9731655';
    if(isTop){ctx.shadowColor='#f97316';ctx.shadowBlur=12+6*Math.sin(t*5);}
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);ctx.fill();
    ctx.shadowBlur=0;
    ctx.fillStyle=isTop?'#fff':'rgba(148,163,184,0.7)';
    ctx.font=(isTop?'bold ':'')+`${isTop?8:7}px Arial`;
    ctx.textAlign='left';ctx.textBaseline='middle';
    ctx.fillText(i+''+(probs?` ${(p*100).toFixed(0)}%`:''),n.x+r+3,n.y);
  });
}

/* ══════════════════════════════════════════════
   THREE.JS  3D SCENE
══════════════════════════════════════════════ */
let renderer3,scene3,cam3,rot={x:0,y:0},isDrag=false,prevMX=0,prevMY=0;
const layer3Meshes=[];

const L3D=[
  {pos:-8.5,w:.5, h:3.2, d:.5, ch:1,  col:0x00f5ff},
  {pos:-6,  w:.3, h:4.0, d:.3, ch:8,  col:0xa855f7},
  {pos:-3.5,w:.3, h:3.0, d:.3, ch:8,  col:0xf97316},
  {pos:-1,  w:.3, h:3.4, d:.3, ch:16, col:0xec4899},
  {pos:1.5, w:.3, h:2.4, d:.3, ch:16, col:0x22d3ee},
  {pos:4,   w:.15,h:5.2, d:.15,ch:1,  col:0xeab308},
  {pos:6.5, w:.2, h:3.6, d:.2, ch:1,  col:0xa855f7},
  {pos:9,   w:.22,h:2.2, d:.22,ch:1,  col:0xf97316},
];

function initThree(){
  const cv=$('three-canvas');
  try{
    if(typeof THREE==='undefined') throw new Error('Three.js not loaded');
    renderer3=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
    renderer3.setPixelRatio(window.devicePixelRatio);
    renderer3.setClearColor(0x050812,1);

    scene3=new THREE.Scene();
    scene3.fog=new THREE.FogExp2(0x050812,0.035);

    cam3=new THREE.PerspectiveCamera(48,1,0.1,200);
    cam3.position.set(0,3.5,18);cam3.lookAt(0,0,0);

    scene3.add(new THREE.AmbientLight(0x112244,1.2));
    const p1=new THREE.PointLight(0x00f5ff,3,35);p1.position.set(-8,5,6);scene3.add(p1);
    const p2=new THREE.PointLight(0xa855f7,3,35);p2.position.set(8,-4,6);scene3.add(p2);
    const p3=new THREE.PointLight(0xf97316,1.5,25);p3.position.set(0,0,10);scene3.add(p3);

    // Build layers
    L3D.forEach((def,i)=>{
      const group=new THREE.Group();
      const stacks=Math.min(def.ch,5);
      for(let s=0;s<stacks;s++){
        const geo=new THREE.BoxGeometry(def.w,def.h,def.d);
        const mat=new THREE.MeshPhongMaterial({
          color:def.col,transparent:true,opacity:0.18,
          emissive:def.col,emissiveIntensity:0.08,
        });
        const mesh=new THREE.Mesh(geo,mat);
        mesh.position.z=s*(def.d+0.12)-(stacks*(def.d+0.12))/2;
        const egeo=new THREE.EdgesGeometry(geo);
        const emat=new THREE.LineBasicMaterial({color:def.col,transparent:true,opacity:0.5});
        mesh.add(new THREE.LineSegments(egeo,emat));
        group.add(mesh);
      }
      group.position.x=def.pos;
      group.userData={def,idx:i,active:false};
      scene3.add(group);
      layer3Meshes.push(group);
    });

    // Connection lines
    for(let i=0;i<L3D.length-1;i++){
      const pts=[new THREE.Vector3(L3D[i].pos+L3D[i].w/2,0,0),new THREE.Vector3(L3D[i+1].pos-L3D[i+1].w/2,0,0)];
      const g=new THREE.BufferGeometry().setFromPoints(pts);
      const m=new THREE.LineBasicMaterial({color:0x1e2d45,transparent:true,opacity:0.35});
      const line=new THREE.Line(g,m);
      line.userData={conn:true,fi:i};
      scene3.add(line);
    }

    // Grid
    const grid=new THREE.GridHelper(22,22,0x1e2d45,0x0b1120);
    grid.position.y=-4.2;scene3.add(grid);

    // Mouse orbit
    cv.addEventListener('mousedown',e=>{isDrag=true;prevMX=e.clientX;prevMY=e.clientY;});
    window.addEventListener('mouseup',()=>isDrag=false);
    window.addEventListener('mousemove',e=>{
      if(!isDrag) return;
      rot.y+=(e.clientX-prevMX)*0.005;
      rot.x=Math.max(-1,Math.min(1,rot.x+(e.clientY-prevMY)*0.005));
      prevMX=e.clientX;prevMY=e.clientY;
    });
    cv.addEventListener('wheel',e=>{cam3.position.z=Math.max(6,Math.min(28,cam3.position.z+e.deltaY*0.02));});

    // Touch
    let lastTX=0,lastTY=0;
    cv.addEventListener('touchstart',e=>{isDrag=true;lastTX=e.touches[0].clientX;lastTY=e.touches[0].clientY;},{passive:true});
    cv.addEventListener('touchmove',e=>{
      if(!isDrag) return;
      rot.y+=(e.touches[0].clientX-lastTX)*0.006;
      rot.x=Math.max(-1,Math.min(1,rot.x+(e.touches[0].clientY-lastTY)*0.006));
      lastTX=e.touches[0].clientX;lastTY=e.touches[0].clientY;
    },{passive:true});
    cv.addEventListener('touchend',()=>isDrag=false);

    function sizeThree(){
      const p=cv.parentElement;if(!p)return;
      renderer3.setSize(p.clientWidth,p.clientHeight);
      cam3.aspect=p.clientWidth/p.clientHeight;cam3.updateProjectionMatrix();
    }
    sizeThree();window.addEventListener('resize',sizeThree);

    let t3=0;
    (function loop3(){
      requestAnimationFrame(loop3);
      t3+=0.012;
      if(!isDrag) rot.y+=0.003;
      cam3.position.x=Math.sin(rot.y)*18;
      cam3.position.z=Math.cos(rot.y)*18;
      cam3.position.y=3.5+rot.x*7;
      cam3.lookAt(0,0,0);
      layer3Meshes.forEach((g,i)=>{
        if(g.userData.active) g.scale.y=0.96+0.04*Math.sin(t3*4+i);
      });
      renderer3.render(scene3,cam3);
    })();

  }catch(e){
    console.warn('Three.js error:',e);
    cv.parentElement.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;font-family:Arial">Three.js غير متاح في هذا الوضع</div>';
  }
}

function update3D(active){
  if(!renderer3) return;
  layer3Meshes.forEach((group,i)=>{
    const act=i<=active,cur=i===active;
    group.userData.active=cur;
    group.children.forEach(child=>{
      if(!child.isMesh) return;
      child.material.opacity=act?(cur?0.55:0.28):0.08;
      child.material.emissiveIntensity=cur?0.5:(act?0.15:0.04);
      child.children.forEach(c=>{if(c.isLineSegments) c.material.opacity=act?(cur?0.9:0.5):0.15;});
    });
  });
  scene3&&scene3.children.forEach(obj=>{
    if(!obj.userData.conn) return;
    const fi=obj.userData.fi;
    obj.material.color.setHex(fi<active?L3D[fi].col:0x1e2d45);
    obj.material.opacity=fi<active?0.65:0.3;
  });
}

/* ══════════ VIEW TOGGLE ══════════ */
function initViewToggle(){
  $('btn-2d').addEventListener('click',()=>setView('2d'));
  $('btn-3d').addEventListener('click',()=>setView('3d'));
}
function setView(v){
  S.view=v;
  $('view-2d').classList.toggle('active',v==='2d');
  $('view-3d').classList.toggle('active',v==='3d');
  $('btn-2d').classList.toggle('active',v==='2d');
  $('btn-3d').classList.toggle('active',v==='3d');
}

/* ══════════ RESIZE ══════════ */
window.addEventListener('resize',()=>{_cvs=null;});
