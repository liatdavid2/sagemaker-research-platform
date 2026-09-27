import React,{useEffect,useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API="/api";

function App(){
  const [catalog,setCatalog]=useState([]);
  const [datasets,setDatasets]=useState([]);
  const [jobs,setJobs]=useState([]);
  const [code,setCode]=useState({});
  const [catalogKey,setCatalogKey]=useState("cifar10");
  const [publicVersion,setPublicVersion]=useState("v3");
  const [datasetId,setDatasetId]=useState("");
  const [framework,setFramework]=useState("pytorch");
  const [epochs,setEpochs]=useState(5);
  const [workers,setWorkers]=useState(1);
  const [maxMinutes,setMaxMinutes]=useState(5);
  const [budget,setBudget]=useState(1);
  const [plannedRuns,setPlannedRuns]=useState(5);
  const [estimate,setEstimate]=useState(null);

  const [uploadName,setUploadName]=useState("");
  const [uploadVersion,setUploadVersion]=useState("v1");
  const [uploadFile,setUploadFile]=useState(null);

  async function refresh(){
    const [cat,d,j,c]=await Promise.all([
      fetch(`${API}/catalog`), fetch(`${API}/datasets`), fetch(`${API}/jobs`), fetch(`${API}/code-version`)
    ]);
    if(cat.ok){const x=await cat.json();setCatalog(x.datasets||[])}
    if(d.ok){
      const x=await d.json(); setDatasets(x.datasets||[]);
      if(!datasetId && x.datasets?.length) setDatasetId(x.datasets[0].id);
    }
    if(j.ok)setJobs(await j.json());
    if(c.ok)setCode(await c.json());
  }

  async function refreshEstimate(){
    const q=new URLSearchParams({
      instance_count:workers,max_minutes:maxMinutes,total_budget_usd:budget,planned_runs:plannedRuns
    });
    const r=await fetch(`${API}/budget-estimate?${q}`);
    if(r.ok)setEstimate(await r.json());
  }

  useEffect(()=>{refresh();const t=setInterval(refresh,3000);return()=>clearInterval(t)},[]);
  useEffect(()=>{refreshEstimate()},[workers,maxMinutes,budget,plannedRuns]);

  async function post(path){
    const r=await fetch(`${API}${path}`,{method:"POST"});
    const body=await r.json();
    if(!r.ok)alert(body.detail||JSON.stringify(body));
    refresh();
  }

  async function downloadAndRegister(){
    const fd=new FormData();
    fd.append("catalog_key",catalogKey);
    fd.append("version",publicVersion);
    const r=await fetch(`${API}/datasets/register-public`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok)alert(body.detail||JSON.stringify(body));
    refresh();
  }

  async function manualUpload(){
    if(!uploadName||!uploadFile){alert("Choose name and file");return}
    const fd=new FormData();
    fd.append("name",uploadName);fd.append("version",uploadVersion);fd.append("dataset",uploadFile);
    const r=await fetch(`${API}/datasets/upload`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok){alert(body.detail||JSON.stringify(body));return}
    setDatasetId(body.id);refresh();
  }

  async function runTraining(){
    const fd=new FormData();
    [["framework",framework],["dataset_id",datasetId],["epochs",epochs],["instance_count",workers],
     ["max_minutes",maxMinutes],["total_budget_usd",budget],["planned_runs",plannedRuns]]
      .forEach(([k,v])=>fd.append(k,v));
    const r=await fetch(`${API}/run-sagemaker`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok)alert(body.detail||JSON.stringify(body));
    refresh();
  }

  const selected=datasets.find(d=>d.id===datasetId);

  return <main>
    <header>
      <div>
        <h1>SageMaker Research Platform</h1>
        <p>Versioned datasets + committed code + controlled parameters + automatic MLflow tracking.</p>
      </div>
      <div className="badge">Reproducible ML</div>
    </header>

    <section className="card">
      <h2>1. Platform</h2>
      <button onClick={()=>post("/setup")}>Setup AWS Resources</button>
      <button className="danger" onClick={()=>confirm("Destroy AWS resources?")&&post("/destroy")}>Destroy AWS Resources</button>
      <a className="linkButton" href="http://localhost:5050" target="_blank" rel="noreferrer">Open MLflow</a>
    </section>

    <section className="card">
      <h2>2. Dataset Registry</h2>
      <p>Preferred flow: choose a trusted public dataset, download it once, hash it, store the immutable source archive in S3, and reuse that exact version.</p>

      <div className="catalogBox">
        <h3>Public Dataset Catalog</h3>
        <div className="budgetGrid">
          <label>Dataset
            <select value={catalogKey} onChange={e=>setCatalogKey(e.target.value)}>
              {catalog.map(d=><option key={d.key} value={d.key}>{d.name}</option>)}
            </select>
          </label>
          <label>Registry version
            <input value={publicVersion} onChange={e=>setPublicVersion(e.target.value)}/>
          </label>
          <div className="actionCell">
            <button onClick={downloadAndRegister}>Download & Register to S3</button>
          </div>
        </div>
      </div>

      <details>
        <summary>Advanced: upload a private/local dataset</summary>
        <div className="budgetGrid">
          <label>Name<input value={uploadName} onChange={e=>setUploadName(e.target.value)}/></label>
          <label>Version<input value={uploadVersion} onChange={e=>setUploadVersion(e.target.value)}/></label>
          <label>File<input type="file" onChange={e=>setUploadFile(e.target.files[0])}/></label>
        </div>
        <button onClick={manualUpload}>Upload Dataset to S3</button>
      </details>

      <h3>Registered datasets</h3>
      {datasets.length===0?<p>No datasets registered yet.</p>:datasets.map(d=><div className="datasetRow" key={d.id}>
        <b>{d.name}-{d.version}</b>
        <div className="small">{d.source_type} · {d.filename}</div>
        <div className="small">SHA256 {d.sha256.slice(0,20)}…</div>
      </div>)}
    </section>

    <section className="card">
      <h2>3. Training Configuration</h2>
      <div className="configSummary">
        <div><span>Dataset</span><b>{selected?`${selected.name}-${selected.version}`:"—"}</b></div>
        <div><span>Code version</span><b className="mono">{code.git_commit?.slice(0,12)||"—"}</b></div>
        <div><span>GPU</span><b>g4dn.xlarge</b></div>
        <div><span>Epochs</span><b>{epochs}</b></div>
      </div>

      {code.dirty&&<p className="bad">Uncommitted changes detected. Commit them before a reproducible cloud run.</p>}

      <div className="budgetGrid">
        <label>Registered dataset
          <select value={datasetId} onChange={e=>setDatasetId(e.target.value)}>
            <option value="">Select dataset</option>
            {datasets.map(d=><option key={d.id} value={d.id}>{d.name}-{d.version}</option>)}
          </select>
        </label>
        <label>Framework
          <select value={framework} onChange={e=>setFramework(e.target.value)}>
            <option value="pytorch">PyTorch / DDP</option>
            <option value="tensorflow">TensorFlow / MultiWorker</option>
          </select>
        </label>
        <label>Epochs
          <input type="number" min="1" max="50" value={epochs} onChange={e=>setEpochs(e.target.value)}/>
        </label>
      </div>

      <div className="seg">
        {[1,2,4].map(n=><button key={n} className={workers===n?"active":""} onClick={()=>setWorkers(n)}>{n} GPU worker{n>1?"s":""}</button>)}
      </div>
    </section>

    <section className="card">
      <h2>4. Budget Guardrail</h2>
      <div className="budgetGrid">
        <label>Total budget ($)<input type="number" step=".1" value={budget} onChange={e=>setBudget(e.target.value)}/></label>
        <label>Planned runs<input type="number" min="1" value={plannedRuns} onChange={e=>setPlannedRuns(e.target.value)}/></label>
        <label>Max runtime / run (min)<input type="number" min="1" max="60" value={maxMinutes} onChange={e=>setMaxMinutes(e.target.value)}/></label>
      </div>
      {estimate&&<div className="budgetSummary">
        Per-run target <b>${estimate.per_run_budget_usd.toFixed(2)}</b> ·
        Estimate <b>${estimate.estimated_run_cost_usd.toFixed(2)}</b>
        <span className={estimate.within_target?"good":"bad"}>{estimate.within_target?" · within target":" · above target"}</span>
      </div>}
    </section>

    <section className="card heroRun">
      <h2>5. Run Training</h2>
      <pre>{`Dataset: ${selected?selected.name+"-"+selected.version:"—"}
Dataset SHA256: ${selected?selected.sha256.slice(0,16)+"…":"—"}
Code version: git commit ${code.git_commit?.slice(0,12)||"—"}
GPU: ${workers} × g4dn.xlarge
Epochs: ${epochs}
Framework: ${framework}
MLflow: automatic`}</pre>
      <button className="runButton"
        disabled={!datasetId||code.dirty||(estimate&&!estimate.within_target)}
        onClick={runTraining}>Run Training</button>
    </section>

    <section className="card">
      <h2>Run History</h2>
      {jobs.length===0?<p>No runs yet.</p>:jobs.map(j=><div className="job" key={j.id}>
        <div><b>{j.type}</b> <span className={`status ${j.status}`}>{j.status}</span></div>
        {j.dataset&&<div className="small">{j.dataset.name}:{j.dataset.version} · {j.framework||""} · {j.instance_count||""} GPU · {j.epochs||""} epochs</div>}
        {j.mlflow_run_id&&<div className="good">MLflow run: {j.mlflow_run_id.slice(0,12)}</div>}
        <pre>{(j.logs||[]).slice(-12).join("\n")}</pre>
      </div>)}
    </section>
  </main>
}
createRoot(document.getElementById("root")).render(<App/>);
