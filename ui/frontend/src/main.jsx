import React,{useEffect,useMemo,useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API="/api";

function App(){
  const [jobs,setJobs]=useState([]);
  const [framework,setFramework]=useState("pytorch");
  const [count,setCount]=useState(1);
  const [zip,setZip]=useState(null);
  const [dataset,setDataset]=useState(null);

  const [budget,setBudget]=useState(1);
  const [plannedRuns,setPlannedRuns]=useState(5);
  const [maxMinutes,setMaxMinutes]=useState(5);
  const [estimate,setEstimate]=useState(null);

  async function refresh(){
    const r=await fetch(`${API}/jobs`);
    if(r.ok)setJobs(await r.json());
  }

  async function refreshEstimate(){
    const q=new URLSearchParams({
      instance_count:String(count),
      max_minutes:String(maxMinutes),
      total_budget_usd:String(budget),
      planned_runs:String(plannedRuns)
    });
    const r=await fetch(`${API}/budget-estimate?${q}`);
    if(r.ok)setEstimate(await r.json());
  }

  useEffect(()=>{refresh();const t=setInterval(refresh,2500);return()=>clearInterval(t)},[]);
  useEffect(()=>{refreshEstimate()},[count,maxMinutes,budget,plannedRuns]);

  async function post(path){
    const r=await fetch(`${API}${path}`,{method:"POST"});
    const j=await r.json();
    if(!r.ok) alert(JSON.stringify(j));
    refresh();
  }

  async function run(mode){
    if(!zip){alert("Choose a researcher ZIP.");return}
    const fd=new FormData();
    fd.append("framework",framework);
    fd.append("max_minutes",maxMinutes);
    if(mode==="sagemaker"){
      fd.append("instance_count",count);
      fd.append("total_budget_usd",budget);
      fd.append("planned_runs",plannedRuns);
    }
    fd.append("training_zip",zip);
    if(dataset)fd.append("dataset",dataset);

    const r=await fetch(`${API}/run-${mode}`,{method:"POST",body:fd});
    const j=await r.json();
    if(!r.ok) alert(typeof j.detail==="string"?j.detail:JSON.stringify(j));
    refresh();
  }

  const perRun=(Number(budget)/Math.max(1,Number(plannedRuns)||1));

  return <main>
    <header>
      <div>
        <h1>SageMaker Research Platform</h1>
        <p>Validate locally first, then run temporary Managed Spot training on SageMaker.</p>
      </div>
      <div className="badge">Default target: $1 / 5 runs</div>
    </header>

    <section className="card">
      <h2>Budget mode</h2>
      <p>These are planning parameters and can be changed. AWS billing is not guaranteed by the estimate because Spot pricing and startup behavior can vary.</p>
      <div className="budgetGrid">
        <label>Total budget target (USD)
          <input type="number" min="0.01" step="0.10" value={budget} onChange={e=>setBudget(e.target.value)}/>
        </label>
        <label>Planned runs
          <input type="number" min="1" max="100" value={plannedRuns} onChange={e=>setPlannedRuns(e.target.value)}/>
        </label>
        <label>Max runtime per run (minutes)
          <input type="number" min="1" max="60" value={maxMinutes} onChange={e=>setMaxMinutes(e.target.value)}/>
        </label>
      </div>

      <div className="budgetSummary">
        <b>Per-run target:</b> ${perRun.toFixed(2)}
        {estimate && <>
          <span> · <b>Planning estimate:</b> ${estimate.estimated_run_cost_usd.toFixed(2)}</span>
          <span className={estimate.within_target?"good":"bad"}>
            {estimate.within_target ? " · within target" : " · above target"}
          </span>
        </>}
      </div>
    </section>

    <section className="card">
      <h2>Platform automation</h2>
      <p>GitHub Actions validates Python, Terraform, React and Docker. AWS setup creates only S3 + IAM; no GPU stays running.</p>
      <button onClick={()=>post("/setup")}>Setup AWS Resources</button>
      <button className="danger" onClick={()=>confirm("Destroy AWS resources?")&&post("/destroy")}>Destroy AWS Resources</button>
    </section>

    <section className="grid">
      <div className="card">
        <h2>Researcher package</h2>
        <label>Framework
          <select value={framework} onChange={e=>setFramework(e.target.value)}>
            <option value="pytorch">PyTorch</option>
            <option value="sklearn">scikit-learn</option>
            <option value="tensorflow">TensorFlow</option>
          </select>
        </label>
        <label>Training ZIP
          <input type="file" accept=".zip" onChange={e=>setZip(e.target.files[0])}/>
        </label>
        <label>Dataset (optional)
          <input type="file" onChange={e=>setDataset(e.target.files[0])}/>
        </label>
        <button onClick={()=>run("local")}>Run Locally First</button>
      </div>

      <div className="card">
        <h2>SageMaker Managed Spot</h2>
        <p>Same researcher package, temporary GPU compute only.</p>
        <div className="seg">
          {[1,2,4].map(n=><button key={n} className={count===n?"active":""} onClick={()=>setCount(n)}>{n} GPU{n>1?"s":""}</button>)}
        </div>
        <p><code>ml.g4dn.xlarge</code> · NVIDIA T4 · Managed Spot only</p>
        <p>Maximum training runtime: <b>{maxMinutes} min</b></p>
        <button disabled={estimate && !estimate.within_target} onClick={()=>run("sagemaker")}>Run on SageMaker</button>
        {estimate && !estimate.within_target &&
          <p className="bad">Current selection is above the configured per-run budget target.</p>
        }
      </div>
    </section>

    <section className="card">
      <h2>Included real-data examples</h2>
      <p><b>PyTorch:</b> CIFAR-10, using only a small subset for a short run.</p>
      <p><b>scikit-learn:</b> RandomForest on Wisconsin Diagnostic Breast Cancer.</p>
      <p><b>TensorFlow:</b> Keras network on Wisconsin Diagnostic Breast Cancer.</p>
      <p>Ready ZIPs: <code>researcher_job_zips/</code></p>
      <p>Real CSV example: <code>sample_data/wisconsin_breast_cancer.csv</code></p>
    </section>

    <section className="card">
      <h2>Researcher ZIP contract</h2>
      <pre>{`research_job.zip
├── train.py             required
├── requirements.txt     optional
├── config.yaml          optional
└── README.md            optional`}</pre>
      <p>Local mode: <code>LOCAL_DATA_DIR</code>, <code>LOCAL_MODEL_DIR</code>. SageMaker: <code>SM_CHANNEL_TRAINING</code>, <code>SM_MODEL_DIR</code>.</p>
    </section>

    <section className="card">
      <h2>Live run status</h2>
      {jobs.length===0?<p>No runs yet.</p>:jobs.map(j=><div className="job" key={j.id}>
        <div><b>{j.type}</b> · {j.name||""} <span className={`status ${j.status}`}>{j.status}</span></div>
        {j.budget && <div className="small">Estimate: ${j.budget.estimated_run_cost_usd} · target/run: ${j.budget.per_run_budget_usd}</div>}
        <pre>{(j.logs||[]).slice(-14).join("\n")}</pre>
      </div>)}
    </section>
  </main>
}
createRoot(document.getElementById("root")).render(<App/>);
