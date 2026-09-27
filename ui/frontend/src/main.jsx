import React, {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API="/api";

function App(){
  const [datasets,setDatasets]=useState([]);
  const [jobs,setJobs]=useState([]);
  const [code,setCode]=useState({});
  const [datasetId,setDatasetId]=useState("");
  const [framework,setFramework]=useState("pytorch");
  const [epochs,setEpochs]=useState(5);
  const [workers,setWorkers]=useState(1);
  const [maxMinutes,setMaxMinutes]=useState(5);
  const [budget,setBudget]=useState(1);
  const [plannedRuns,setPlannedRuns]=useState(5);
  const [estimate,setEstimate]=useState(null);

  const [datasetName,setDatasetName]=useState("CIFAR-10");
  const [datasetVersion,setDatasetVersion]=useState("v3");
  const [datasetFile,setDatasetFile]=useState(null);

  async function refresh(){
    const [d,j,c]=await Promise.all([
      fetch(`${API}/datasets`), fetch(`${API}/jobs`), fetch(`${API}/code-version`)
    ]);
    if(d.ok){
      const data=await d.json();
      setDatasets(data.datasets||[]);
      if(!datasetId && data.datasets?.length) setDatasetId(data.datasets[0].id);
    }
    if(j.ok) setJobs(await j.json());
    if(c.ok) setCode(await c.json());
  }

  async function refreshEstimate(){
    const q=new URLSearchParams({
      instance_count:workers,
      max_minutes:maxMinutes,
      total_budget_usd:budget,
      planned_runs:plannedRuns
    });
    const r=await fetch(`${API}/budget-estimate?${q}`);
    if(r.ok) setEstimate(await r.json());
  }

  useEffect(()=>{refresh();const t=setInterval(refresh,3000);return()=>clearInterval(t)},[]);
  useEffect(()=>{refreshEstimate()},[workers,maxMinutes,budget,plannedRuns]);

  async function post(path){
    const r=await fetch(`${API}${path}`,{method:"POST"});
    const body=await r.json();
    if(!r.ok) alert(body.detail||JSON.stringify(body));
    refresh();
  }

  async function uploadDataset(){
    if(!datasetFile){alert("Choose a dataset file");return}
    const fd=new FormData();
    fd.append("name",datasetName);
    fd.append("version",datasetVersion);
    fd.append("dataset",datasetFile);
    const r=await fetch(`${API}/datasets/upload`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok){alert(body.detail||JSON.stringify(body));return}
    setDatasetId(body.id);
    refresh();
  }

  async function runTraining(){
    const fd=new FormData();
    fd.append("framework",framework);
    fd.append("dataset_id",datasetId);
    fd.append("epochs",epochs);
    fd.append("instance_count",workers);
    fd.append("max_minutes",maxMinutes);
    fd.append("total_budget_usd",budget);
    fd.append("planned_runs",plannedRuns);
    const r=await fetch(`${API}/run-sagemaker`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok) alert(body.detail||JSON.stringify(body));
    refresh();
  }

  const selected=datasets.find(d=>d.id===datasetId);

  return <main>
    <header>
      <div>
        <h1>SageMaker Research Platform</h1>
        <p>Versioned datasets + committed code + controlled parameters + MLflow tracking.</p>
      </div>
      <div className="badge">Reproducible ML</div>
    </header>

    <section className="card">
      <h2>Platform</h2>
      <button onClick={()=>post("/setup")}>Setup AWS Resources</button>
      <button className="danger" onClick={()=>confirm("Destroy AWS resources?")&&post("/destroy")}>Destroy AWS Resources</button>
      <a className="linkButton" href="http://localhost:5050" target="_blank" rel="noreferrer">Open MLflow</a>
    </section>

    <section className="card">
      <h2>Dataset Registry</h2>
      <p>Upload a dataset once to S3 with a logical version. Training runs select an existing version.</p>
      <div className="budgetGrid">
        <label>Dataset name<input value={datasetName} onChange={e=>setDatasetName(e.target.value)}/></label>
        <label>Version<input value={datasetVersion} onChange={e=>setDatasetVersion(e.target.value)}/></label>
        <label>Dataset file<input type="file" onChange={e=>setDatasetFile(e.target.files[0])}/></label>
      </div>
      <button onClick={uploadDataset}>Upload Dataset to S3</button>

      <div className="registry">
        {datasets.map(d=><div className="datasetRow" key={d.id}>
          <b>{d.name}-{d.version}</b>
          <div className="small">{d.filename} · SHA256 {d.sha256.slice(0,16)}…</div>
        </div>)}
      </div>
    </section>

    <section className="card">
      <h2>Training Configuration</h2>
      <div className="configSummary">
        <div><span>Dataset</span><b>{selected?`${selected.name}-${selected.version}`:"—"}</b></div>
        <div><span>Code version</span><b className="mono">{code.git_commit?.slice(0,12)||"—"}</b></div>
        <div><span>GPU</span><b>g4dn.xlarge</b></div>
        <div><span>Epochs</span><b>{epochs}</b></div>
      </div>

      {code.dirty && <p className="bad">Uncommitted changes detected. Commit them before running so the experiment is reproducible.</p>}

      <div className="budgetGrid">
        <label>Dataset
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
        <label>Epochs<input type="number" min="1" max="50" value={epochs} onChange={e=>setEpochs(e.target.value)}/></label>
      </div>

      <div className="seg">
        {[1,2,4].map(n=><button key={n} className={workers===n?"active":""} onClick={()=>setWorkers(n)}>{n} GPU worker{n>1?"s":""}</button>)}
      </div>
    </section>

    <section className="card">
      <h2>Budget Guardrail</h2>
      <div className="budgetGrid">
        <label>Total budget ($)<input type="number" step=".1" value={budget} onChange={e=>setBudget(e.target.value)}/></label>
        <label>Planned runs<input type="number" min="1" value={plannedRuns} onChange={e=>setPlannedRuns(e.target.value)}/></label>
        <label>Max runtime / run (min)<input type="number" min="1" max="60" value={maxMinutes} onChange={e=>setMaxMinutes(e.target.value)}/></label>
      </div>
      {estimate && <div className="budgetSummary">
        Per-run target <b>${estimate.per_run_budget_usd.toFixed(2)}</b> ·
        Estimate <b>${estimate.estimated_run_cost_usd.toFixed(2)}</b>
        <span className={estimate.within_target?"good":"bad"}>
          {estimate.within_target?" · within target":" · above target"}
        </span>
      </div>}
    </section>

    <section className="card heroRun">
      <h2>Run Training</h2>
      <pre>{`Dataset: ${selected?selected.name+"-"+selected.version:"—"}
Code version: git commit ${code.git_commit?.slice(0,12)||"—"}
GPU: ${workers} × g4dn.xlarge
Epochs: ${epochs}
Framework: ${framework}
MLflow: automatic`}</pre>

      <button
        className="runButton"
        disabled={!datasetId || code.dirty || (estimate && !estimate.within_target)}
        onClick={runTraining}>
        Run Training
      </button>
    </section>

    <section className="card">
      <h2>Run History</h2>
      {jobs.length===0?<p>No runs yet.</p>:jobs.map(j=><div className="job" key={j.id}>
        <div><b>{j.type}</b> <span className={`status ${j.status}`}>{j.status}</span></div>
        {j.dataset && <div className="small">
          {j.dataset.name}:{j.dataset.version} · {j.framework} · {j.instance_count} GPU ·
          {j.epochs} epochs · commit {j.git_commit?.slice(0,8)}
        </div>}
        {j.mlflow_run_id && <div className="good">MLflow run: {j.mlflow_run_id.slice(0,12)}</div>}
        <pre>{(j.logs||[]).slice(-12).join("\n")}</pre>
      </div>)}
    </section>
  </main>
}

createRoot(document.getElementById("root")).render(<App/>);
