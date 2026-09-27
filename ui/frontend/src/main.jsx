import React,{useEffect,useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API="/api";

function App(){
  const [catalog,setCatalog]=useState([]);
  const [datasets,setDatasets]=useState([]);
  const [codes,setCodes]=useState([]);
  const [jobs,setJobs]=useState([]);
  const [platform,setPlatform]=useState({ready:false,latest_setup_job:null,bucket:null,role:null});

  const [catalogKey,setCatalogKey]=useState("cifar10");
  const [publicVersion,setPublicVersion]=useState("v3");
  const [datasetJobId,setDatasetJobId]=useState(null);

  const [codeName,setCodeName]=useState("vision-cnn");
  const [codeVersion,setCodeVersion]=useState("v1");
  const [codeFramework,setCodeFramework]=useState("pytorch");
  const [codeFile,setCodeFile]=useState(null);
  const [codeUploading,setCodeUploading]=useState(false);
  const [codeUploadMessage,setCodeUploadMessage]=useState("");

  const [datasetId,setDatasetId]=useState("");
  const [codeId,setCodeId]=useState("");
  const [framework,setFramework]=useState("pytorch");
  const [epochs,setEpochs]=useState(5);
  const [workers,setWorkers]=useState(1);
  const [maxMinutes,setMaxMinutes]=useState(5);
  const [budget,setBudget]=useState(1);
  const [plannedRuns,setPlannedRuns]=useState(5);
  const [estimate,setEstimate]=useState(null);

  const [useSubset,setUseSubset]=useState(true);
  const [trainMaxPerClass,setTrainMaxPerClass]=useState(200);
  const [testMaxPerClass,setTestMaxPerClass]=useState(50);
  const [subsetSeed,setSubsetSeed]=useState(42);

  const [trainingJobId,setTrainingJobId]=useState(null);

  async function refresh(){
    const [cat,d,c,j,p]=await Promise.all([
      fetch(`${API}/catalog`),fetch(`${API}/datasets`),fetch(`${API}/codes`),fetch(`${API}/jobs`),
      fetch(`${API}/platform-status`)
    ]);
    if(cat.ok){const x=await cat.json();setCatalog(x.datasets||[])}
    if(d.ok){
      const x=await d.json();setDatasets(x.datasets||[]);
      if(!datasetId&&x.datasets?.length)setDatasetId(x.datasets[0].id);
    }
    if(c.ok){
      const x=await c.json();setCodes(x.codes||[]);
      if(!codeId&&x.codes?.length)setCodeId(x.codes[0].id);
    }
    if(j.ok)setJobs(await j.json());
    if(p.ok)setPlatform(await p.json());
  }

  async function refreshEstimate(){
    const q=new URLSearchParams({
      instance_count:workers,max_minutes:maxMinutes,total_budget_usd:budget,planned_runs:plannedRuns
    });
    const r=await fetch(`${API}/budget-estimate?${q}`);
    if(r.ok)setEstimate(await r.json());
  }

  useEffect(()=>{refresh();const t=setInterval(refresh,1500);return()=>clearInterval(t)},[]);
  useEffect(()=>{refreshEstimate()},[workers,maxMinutes,budget,plannedRuns]);

  async function post(path){
    const r=await fetch(`${API}${path}`,{method:"POST"});
    const body=await r.json();
    if(!r.ok)alert(body.detail||JSON.stringify(body));
    refresh();
  }

  async function downloadDataset(){
    const fd=new FormData();
    fd.append("catalog_key",catalogKey);
    fd.append("version",publicVersion);
    const r=await fetch(`${API}/datasets/register-public`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok){alert(body.detail||JSON.stringify(body));return}
    setDatasetJobId(body.id);
    refresh();
  }

  async function uploadCode(){
    if(!codeFile){alert("Choose training-code.zip");return}
    setCodeUploading(true);setCodeUploadMessage("Uploading training code...");
    const fd=new FormData();
    fd.append("name",codeName);
    fd.append("version",codeVersion);
    fd.append("framework",codeFramework);
    fd.append("code_zip",codeFile);
    const r=await fetch(`${API}/codes/upload`,{method:"POST",body:fd});
    const body=await r.json();
    setCodeUploading(false);
    if(!r.ok){setCodeUploadMessage("");alert(body.detail||JSON.stringify(body));return}
    setCodeUploadMessage(`Registered ${body.name}-${body.version} · SHA256 ${body.sha256.slice(0,12)}…`);
    setCodeId(body.id);
    setFramework(body.framework);
    refresh();
  }

  async function runTraining(){
    const fd=new FormData();
    [["framework",framework],["dataset_id",datasetId],["code_id",codeId],["epochs",epochs],
     ["instance_count",workers],["max_minutes",maxMinutes],["total_budget_usd",budget],
     ["planned_runs",plannedRuns],
     ["use_subset",useSubset],
     ["train_max_per_class",trainMaxPerClass],
     ["test_max_per_class",testMaxPerClass],
     ["subset_seed",subsetSeed]
    ].forEach(([k,v])=>fd.append(k,v));
    const r=await fetch(`${API}/run-sagemaker`,{method:"POST",body:fd});
    const body=await r.json();
    if(!r.ok){alert(body.detail||JSON.stringify(body));return}
    setTrainingJobId(body.id);
    refresh();
  }

  const selectedDataset=datasets.find(d=>d.id===datasetId);
  const setupJob=platform.latest_setup_job;
  const selectedCode=codes.find(c=>c.id===codeId);
  const datasetJob=jobs.find(j=>j.id===datasetJobId);
  const trainingJob=jobs.find(j=>j.id===trainingJobId);

  return <main>
    <header>
      <div>
        <h1>SageMaker Research Platform</h1>
        <p>Dataset Registry + Training Code Registry + SageMaker + automatic MLflow.</p>
      </div>
      <div className="badge">Reproducible ML</div>
    </header>

    <section className="card">
      <h2>1. Platform</h2>

      <div className="platformActions">
        <button
          onClick={()=>post("/setup")}
          disabled={setupJob && ["queued","running"].includes(setupJob.status)}>
          {setupJob && ["queued","running"].includes(setupJob.status) ? "Setting up AWS..." : "Setup AWS Resources"}
        </button>
        <button className="danger" onClick={()=>confirm("Destroy AWS resources?")&&post("/destroy")}>Destroy AWS Resources</button>
        <a className="linkButton" href="http://localhost:5050" target="_blank" rel="noreferrer">Open MLflow</a>
      </div>

      {setupJob && ["queued","running"].includes(setupJob.status) && <div className={`setupStatus ${setupJob.status}`}>
        <div className="progressHeader">
          <span>{setupJob.stage || setupJob.status}</span>
          <b>{setupJob.progress_percent ?? 0}%</b>
        </div>

        <div className="progressTrack">
          <div className="progressFill" style={{width:`${setupJob.progress_percent ?? 0}%`}}></div>
        </div>
      </div>}

      {platform.ready && !(setupJob && ["queued","running"].includes(setupJob.status)) && <div className="setupStatus completed">
        <div className="progressHeader">
          <span>Ready</span>
          <b>100%</b>
        </div>
        <div className="progressTrack">
          <div className="progressFill" style={{width:"100%"}}></div>
        </div>
        <div className="good setupHeadline">AWS Resources Ready ✓</div>
        <div className="small monoBlock">S3 bucket: {platform.bucket}</div>
        <div className="small monoBlock">SageMaker role: {platform.role}</div>
        <div className="small persistentNote">Recovered from Terraform/AWS state — survives browser refresh and backend restart.</div>
      </div>}

      {setupJob?.status==="failed" && !platform.ready && <div className="setupStatus failed">
        <div className="bad setupHeadline">Setup Failed ✕</div>
        <div className="small">{setupJob.logs?.slice(-1)[0]}</div>
      </div>}

      {!platform.ready && !setupJob && <p className="small">AWS resources are not set up yet.</p>}
    </section>

    <section className="card">
      <h2>2. Dataset Registry</h2>
      <p>Download a trusted public dataset once, register the exact archive in S3, and reuse that version.</p>
      {!platform.ready && <p className="warningText">Complete Setup AWS Resources first.</p>}

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
          <button onClick={downloadDataset} disabled={!platform.ready || (datasetJob&&["queued","running"].includes(datasetJob.status))}>
            Download & Register to S3
          </button>
        </div>
      </div>

      {datasetJob && <div className="inlineProgress">
        <div className="progressHeader">
          <span>{datasetJob.stage || "Working..."}</span>
          <b>{datasetJob.progress_percent ?? 0}%</b>
        </div>
        <div className="progressTrack">
          <div className="progressFill" style={{width:`${datasetJob.progress_percent ?? 0}%`}}></div>
        </div>
        <div className="progressMeta">
          <span>
            {datasetJob.total_bytes>0
              ? `${(datasetJob.downloaded_bytes/1024/1024).toFixed(1)} MB / ${(datasetJob.total_bytes/1024/1024).toFixed(1)} MB`
              : datasetJob.cache_hit==="s3" ? "No download needed" : "Preparing..."}
          </span>
          <span>Mode: <b>{datasetJob.download_mode || "Checking..."}</b></span>
          {datasetJob.speed_bps>0 && <span>Speed: <b>{(datasetJob.speed_bps/1024/1024).toFixed(1)} MB/s</b></span>}
          {datasetJob.eta_seconds!=null && datasetJob.eta_seconds>0 && <span>ETA: <b>{datasetJob.eta_seconds}s</b></span>}
        </div>
        {datasetJob.cache_hit==="local" && <div className="good">Local cache hit — internet download skipped.</div>}
        {datasetJob.cache_hit==="s3" && <div className="good">Already registered in S3 — download and upload skipped.</div>}
        {datasetJob.status==="failed"&&<div className="bad">{datasetJob.logs?.slice(-1)[0]}</div>}
      </div>}

      <div className="registry">
        {datasets.map(d=><div className="datasetRow" key={d.id}>
          <b>{d.name}-{d.version}</b>
          <div className="small">{d.filename} · SHA256 {d.sha256.slice(0,16)}…</div>
        </div>)}
      </div>
    </section>

    <section className="card">
      <h2>3. Training Code Registry</h2>
      <p>Researcher uploads a versioned ZIP containing <code>train.py</code>. The platform hashes it, stores it in S3, and uses that exact snapshot for SageMaker.</p>
      {!platform.ready && <p className="warningText">Complete Setup AWS Resources first.</p>}

      <div className="budgetGrid">
        <label>Code name<input value={codeName} onChange={e=>setCodeName(e.target.value)}/></label>
        <label>Version<input value={codeVersion} onChange={e=>setCodeVersion(e.target.value)}/></label>
        <label>Framework
          <select value={codeFramework} onChange={e=>setCodeFramework(e.target.value)}>
            <option value="pytorch">PyTorch</option>
            <option value="tensorflow">TensorFlow</option>
          </select>
        </label>
      </div>

      <label>Training code ZIP
        <input type="file" accept=".zip" onChange={e=>setCodeFile(e.target.files[0])}/>
      </label>

      <button onClick={uploadCode} disabled={!platform.ready || codeUploading}>
        {codeUploading ? "Uploading..." : "Upload & Register Training Code"}
      </button>
      {codeUploadMessage&&<span className="good inlineMessage">{codeUploadMessage}</span>}

      <pre>{`training-code.zip
├── train.py             required
├── requirements.txt     optional
├── config.yaml          optional
└── README.md            optional`}</pre>

      <div className="registry">
        {codes.map(c=><div className="datasetRow" key={c.id}>
          <b>{c.name}-{c.version}</b>
          <div className="small">{c.framework} · {c.filename} · SHA256 {c.sha256.slice(0,16)}…</div>
        </div>)}
      </div>
    </section>

    <section className="card">
      <h2>4. Training Configuration</h2>
      <div className="configSummary">
        <div><span>Dataset</span><b>{selectedDataset?`${selectedDataset.name}-${selectedDataset.version}`:"—"}</b></div>
        <div><span>Training code</span><b>{selectedCode?`${selectedCode.name}-${selectedCode.version}`:"—"}</b></div>
        <div><span>GPU</span><b>g4dn.xlarge</b></div>
        <div><span>Epochs</span><b>{epochs}</b></div>
      </div>

      <div className="budgetGrid">
        <label>Dataset
          <select value={datasetId} onChange={e=>setDatasetId(e.target.value)}>
            <option value="">Select dataset</option>
            {datasets.map(d=><option key={d.id} value={d.id}>{d.name}-{d.version}</option>)}
          </select>
        </label>
        <label>Training code
          <select value={codeId} onChange={e=>{
            const id=e.target.value;setCodeId(id);
            const c=codes.find(x=>x.id===id);if(c)setFramework(c.framework);
          }}>
            <option value="">Select code</option>
            {codes.map(c=><option key={c.id} value={c.id}>{c.name}-{c.version}</option>)}
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

      <div className="subsetBox">
        <label className="checkboxLabel">
          <input type="checkbox" checked={useSubset} onChange={e=>setUseSubset(e.target.checked)}/>
          <span>Use balanced subset</span>
        </label>

        {useSubset && <>
          <p className="small">Take at most N samples from each class. For CIFAR-10, 200 train/class = 2,000 training images.</p>
          <div className="budgetGrid">
            <label>Train max / class
              <input type="number" min="1" max="5000" value={trainMaxPerClass} onChange={e=>setTrainMaxPerClass(e.target.value)}/>
            </label>
            <label>Test max / class
              <input type="number" min="1" max="1000" value={testMaxPerClass} onChange={e=>setTestMaxPerClass(e.target.value)}/>
            </label>
            <label>Subset seed
              <input type="number" value={subsetSeed} onChange={e=>setSubsetSeed(e.target.value)}/>
            </label>
          </div>
          <div className="subsetSummary">
            Approx. train samples: <b>{10 * Number(trainMaxPerClass || 0)}</b> ·
            test samples: <b>{10 * Number(testMaxPerClass || 0)}</b> ·
            seed: <b>{subsetSeed}</b>
          </div>
        </>}
      </div>

      <div className="seg">
        {[1,2,4].map(n=><button key={n} className={workers===n?"active":""} onClick={()=>setWorkers(n)}>{n} GPU worker{n>1?"s":""}</button>)}
      </div>
    </section>

    <section className="card">
      <h2>5. Budget Guardrail</h2>
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
      <h2>6. Run Training</h2>
      <pre>{`Dataset: ${selectedDataset?selectedDataset.name+"-"+selectedDataset.version:"—"}
Dataset SHA256: ${selectedDataset?selectedDataset.sha256.slice(0,16)+"…":"—"}
Training code: ${selectedCode?selectedCode.name+"-"+selectedCode.version:"—"}
Code SHA256: ${selectedCode?selectedCode.sha256.slice(0,16)+"…":"—"}
GPU: ${workers} × g4dn.xlarge
Epochs: ${epochs}
Framework: ${framework}
Subset: ${useSubset ? `balanced · train ${trainMaxPerClass}/class · test ${testMaxPerClass}/class · seed ${subsetSeed}` : "full dataset"}
MLflow: automatic`}</pre>

      <button className="runButton"
        disabled={!platform.ready||!datasetId||!codeId||(estimate&&!estimate.within_target)||trainingJob?.status==="running"}
        onClick={runTraining}>
        Run Training
      </button>

      {trainingJob&&<div className="currentRun">
        <div><b>{trainingJob.stage||trainingJob.status}</b> · <span className={`status ${trainingJob.status}`}>{trainingJob.status}</span></div>
        {trainingJob.mlflow_run_id&&<div className="good">MLflow run: {trainingJob.mlflow_run_id.slice(0,12)}</div>}
        {trainingJob.elapsed_seconds&&<div className="small">Elapsed: {trainingJob.elapsed_seconds}s</div>}
        {trainingJob.status==="failed"&&<div className="bad">{trainingJob.logs?.slice(-1)[0]}</div>}
      </div>}
    </section>
  </main>
}

createRoot(document.getElementById("root")).render(<App/>);
