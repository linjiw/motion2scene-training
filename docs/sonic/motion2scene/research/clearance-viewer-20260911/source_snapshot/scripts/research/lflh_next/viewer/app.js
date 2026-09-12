'use strict';
const $=id=>document.getElementById(id), scene=$('scene');
const names=DATA.motions.map(m=>m.name), keys=Object.keys(DATA.distributions);
names.forEach((name,i)=>{for(const id of ['motion','alternative']){const o=new Option(name,i);$(id).add(o);}});
keys.forEach(k=>$('model').add(new Option(k,k)));
$('motion').value='2';$('model').value='contrast-301';
let selected=0,playing=false,busy=false,last=0;
const camera={eye:{x:1.5,y:-2.2,z:1.1},up:{x:0,y:0,z:1}};
const layout={paper_bgcolor:'#101823',plot_bgcolor:'#101823',font:{color:'#b9cadf'},margin:{l:0,r:0,t:0,b:0},showlegend:false,uirevision:'camera',scene:{aspectmode:'data',camera,
 xaxis:{title:{text:'Forward x (m)'},range:[-.5,5.5],gridcolor:'#2e4055',backgroundcolor:'#162230'},
 yaxis:{title:{text:'Lateral y (m)'},range:[-1.25,1.25],gridcolor:'#2e4055',backgroundcolor:'#162230'},
 zaxis:{title:{text:'Height z (m)'},range:[0,2.2],gridcolor:'#2e4055',backgroundcolor:'#162230'}}};
const pct=x=>(100*x).toFixed(2)+'%';
function current(){const m=+$('motion').value, d=DATA.distributions[$('model').value];return{m,d,q:d[$('mode').value][m]};}
function mesh(m,frame,color){
 const motion=DATA.motions[m],a=motion.a[frame],b=motion.b[frame],rs=motion.radius;
 const x=[],y=[],z=[],ii=[],jj=[],kk=[], rings=[-Math.PI/2,-Math.PI/3,-Math.PI/6,0,0,Math.PI/6,Math.PI/3,Math.PI/2],n=8;
 for(let c=0;c<a.length;c++){
  const av=a[c],bv=b[c];let axis=bv.map((v,k)=>v-av[k]),len=Math.hypot(...axis);axis=len>1e-10?axis.map(v=>v/len):[0,0,1];
  const ref=Math.abs(axis[2])<.9?[0,0,1]:[1,0,0];
  const cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  let u=cross(axis,ref),ul=Math.hypot(...u);u=u.map(v=>v/ul);const v=cross(axis,u),base=x.length;
  rings.forEach((theta,r)=>{const center=r<4?av:bv;for(let j=0;j<n;j++){const phi=j*2*Math.PI/n;
   const p=center.map((value,k)=>value+rs[c]*(Math.sin(theta)*axis[k]+Math.cos(theta)*(Math.cos(phi)*u[k]+Math.sin(phi)*v[k])));x.push(p[0]);y.push(p[1]);z.push(p[2]);
  }});
  for(let r=0;r<rings.length-1;r++)for(let j=0;j<n;j++){const s=base+r*n+j,t=base+r*n+(j+1)%n;ii.push(s,t);jj.push(t,t+n);kk.push(s+n,s+n);}
 }
 return{type:'mesh3d',x,y,z,i:ii,j:jj,k:kk,color,opacity:.85,hoverinfo:'skip',flatshading:false};
}
function beamTrace(){const [cx,under]=DATA.coordinates[selected],h=DATA.half_extents;const x=[],y=[],z=[];
 for(const a of [-1,1])for(const b of [-1,1])for(const c of [-1,1]){x.push(cx+a*h[0]);y.push(b*h[1]);z.push(under+h[2]+c*h[2]);}
 return{type:'mesh3d',x,y,z,i:[0,0,4,4,0,0,2,2,0,0,1,1],j:[1,3,6,7,4,5,3,7,2,6,5,7],k:[3,2,7,5,5,1,7,6,6,4,7,3],color:'#f6b64b',opacity:.32,hoverinfo:'skip',visible:$('beam').checked&&current().q.some(p=>p>0)};}
function detail(){const{m,q}=current(),lo=DATA.lower[m][selected],hi=DATA.upper[m][selected],c=DATA.coordinates[selected];
 const state=DATA.critical[m][selected]?'CRITICAL: target clear, another schedule has a capsule penetration witness':lo>=DATA.margin?'Target capsule clear; no alternative penetration witness':hi<=-DATA.margin?'Target capsule penetration witness':'Uncertain / insufficient clearance margin';
 $('detail').textContent=`Selected beam: x ${c[0].toFixed(3)} m · underside ${c[1].toFixed(3)} m · probability ${pct(q[selected])} · lower gap ${(1000*lo).toFixed(1)} mm · upper gap ${(1000*hi).toFixed(1)} mm. ${state}.`;
 $('clock').textContent=`${$('frame').value} / ${DATA.motions[m].frames-1} (${(+$('frame').value/DATA.motions[m].fps).toFixed(2)} s)`;
}
function traces(){const{m,q}=current(),f=+$('frame').value,motion=DATA.motions[m],alt=+$('alternative').value;
 const probs=$('color').value==='probability';let color=q;
 if(!probs)color=q.map((_,j)=>DATA.critical[m][j]?3:DATA.lower[m][j]>=.01?2:DATA.upper[m][j]<=-.01?0:1);
 const cloud={type:'scatter3d',mode:'markers',x:DATA.coordinates.map(c=>c[0]),y:q.map(()=>0),z:DATA.coordinates.map(c=>c[1]+.1),
  customdata:q.map((_,j)=>j),text:q.map((p,j)=>`Cell ${j} · q=${pct(p)}<br>Lower gap ${(1000*DATA.lower[m][j]).toFixed(1)} mm`),hovertemplate:'%{text}<extra></extra>',
  marker:{size:q.map(p=>2+10*Math.sqrt(p/Math.max(...q,1e-20))),color,cmin:0,cmax:probs?Math.max(...q,1e-20):3,
   colorscale:probs?'Viridis':[[0,'#ed6c72'],[.33,'#97a5b5'],[.67,'#57bde3'],[1,'#49d6c0']],opacity:.8,showscale:probs,colorbar:{title:{text:'Cell probability'},thickness:12,len:.6}}};
 const root={type:'scatter3d',mode:'lines',x:motion.root.map(p=>p[0]),y:motion.root.map(p=>p[1]),z:motion.root.map(p=>p[2]),line:{color:'#49d6c0',width:4},hoverinfo:'skip'};
 const sweep={type:'scatter3d',mode:'markers',x:[],y:[],z:[],marker:{color:'#718a9e',size:1.8,opacity:.18},hoverinfo:'skip',visible:$('ghost').checked};
 for(let t=0;t<motion.frames;t+=12)for(let c=0;c<motion.a[t].length;c++)for(const p of [motion.a[t][c],motion.b[t][c]]){sweep.x.push(p[0]);sweep.y.push(p[1]);sweep.z.push(p[2]);}
 return[cloud,root,sweep,mesh(m,f,'#49d6c0'),beamTrace(),alt>=0?mesh(alt,Math.min(f,DATA.motions[alt].frames-1),'#cf95ed'):{type:'mesh3d',x:[],y:[],z:[],hoverinfo:'skip'}];
}
async function refresh(resetSelection=false){if(busy)return;busy=true;try{
 const{m,d,q}=current();$('frame').max=DATA.motions[m].frames-1;
 if(resetSelection)selected=q.indexOf(Math.max(...q));
 $('critical').textContent=pct(q.reduce((s,p,j)=>s+p*DATA.critical[m][j],0));
 $('collision').textContent=pct(q.reduce((s,p,j)=>s+p*(DATA.upper[m][j]<=-.01),0));$('retained').textContent=pct(d.retained[m]);
 $('note').textContent=q.every(p=>p===0)?'ABSTAIN: no supported placement. No obstacle is generated.':($('mode').value==='constrained'?'Geometry-query projection active; unknown and insufficient-clearance cells receive zero probability. ':'Raw network output; no generation-time clearance rejection. ')+( $('color').value==='geometry'?'Colors: teal critical · blue clear · gray uncertain · red penetration witness.':'Point size and color show probability; all 384 queried cells remain visible.');
 await Plotly.react(scene,traces(),layout,{responsive:true,displaylogo:false});detail();
 }finally{busy=false;}}
async function animateFrame(){if(busy)return;busy=true;try{const{m}=current(),f=+$('frame').value,alt=+$('alternative').value;const body=mesh(m,f,'#49d6c0');await Plotly.restyle(scene,{x:[body.x],y:[body.y],z:[body.z]},[3]);if(alt>=0){const other=mesh(alt,Math.min(f,DATA.motions[alt].frames-1),'#cf95ed');await Plotly.restyle(scene,{x:[other.x],y:[other.y],z:[other.z]},[5]);}detail();}finally{busy=false;}}
for(const id of ['motion','model','mode'])$(id).addEventListener('change',()=>refresh(true));
for(const id of ['color','alternative','ghost','beam'])$(id).addEventListener('change',()=>refresh());
$('frame').addEventListener('input',animateFrame);
$('sample').onclick=()=>{const{q}=current();const total=q.reduce((a,b)=>a+b,0);if(total===0)return;let r=Math.random()*total;selected=q.length-1;for(let j=0;j<q.length;j++){r-=q[j];if(r<=0){selected=j;break;}}refresh();};
$('reset').onclick=()=>Plotly.relayout(scene,{'scene.camera':camera});
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause motion':'Play motion';last=performance.now();};
function tick(now){if(playing&&!busy&&now-last>=100){const steps=Math.max(1,Math.floor((now-last)/1000*DATA.motions[+$('motion').value].fps));last=now;$('frame').value=(+$('frame').value+steps)%(+$('frame').max+1);animateFrame();}requestAnimationFrame(tick);}
refresh(true).then(()=>{scene.on('plotly_click',event=>{const point=event.points[0];if(point.curveNumber===0){selected=point.customdata;refresh();}});window.VIEWER_READY=true;requestAnimationFrame(tick);});
