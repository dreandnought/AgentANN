const stateColors = {
  IDLE: "#6b7280",
  LOADED_WAITING: "#22c55e",
  COMPUTING_FORWARD: "#eab308",
  FORWARD_SENT: "#3b82f6",
  WAITING_BACKWARD_INPUT: "#3b82f6",
  COMPUTING_BACKWARD: "#fb923c",
  UPDATED: "#10b981",
  ERROR: "#ef4444",
}

const agentStates = {}
const agentErrors = {}
const agentWeights = {}
const agentLLMRequests = {}
const agentLLMResponses = {}
let inferenceResult = null

const elStep = document.getElementById("m-step")
const elLoss = document.getElementById("m-loss")
const elAcc = document.getElementById("m-acc")
const elLP = document.getElementById("m-lp")

const elId = document.getElementById("i-id")
const elState = document.getElementById("i-state")
const elErr = document.getElementById("i-err")
const elWJson = document.getElementById("w-json")
const elLogs = document.getElementById("llm-logs")

const inpWeights = document.getElementById("inp-weights")
const btnInfer = document.getElementById("btn-infer")

function buildGraphSpec() {
  const hidden = Array.from({ length: 8 }, (_, i) => `H${i}`)
  const output = Array.from({ length: 3 }, (_, i) => `O${i}`)

  const nodes = [
    { id: "coordinator", name: "Coordinator", x: 0, y: 0, symbolSize: 60 },
    ...hidden.map((id, i) => ({
      id,
      name: id,
      x: 250,
      y: -220 + i * 48,
      symbolSize: 38,
    })),
    ...output.map((id, i) => ({
      id,
      name: id,
      x: 520,
      y: -52 + (i - 1) * 48, // Centered vertically with hidden layer
      symbolSize: 38,
    })),
    {
      id: "Result",
      name: inferenceResult ? `Class ${inferenceResult.prediction}` : "Result",
      x: 750,
      y: -52,
      symbolSize: 50,
    }
  ]

  const links = []
  for (const h of hidden) links.push({ source: "coordinator", target: h })
  for (const h of hidden) for (const o of output) links.push({ source: h, target: o })
  for (const o of output) links.push({ source: o, target: "Result" })
  for (const o of output) links.push({ source: o, target: "coordinator" })

  return { nodes, links }
}

function nodeColor(id) {
  if (id === "Result") return inferenceResult ? stateColors["LOADED_WAITING"] : stateColors["IDLE"]
  const st = agentStates[id]
  if ((!st || st === "IDLE" || st === "WAITING_FORWARD_INPUT") && agentWeights[id]) {
    return stateColors["LOADED_WAITING"]
  }
  if (st === "WAITING_FORWARD_INPUT" && !agentWeights[id]) {
    return stateColors["IDLE"]
  }
  return stateColors[st] || stateColors["IDLE"]
}

const chart = echarts.init(document.getElementById("graph"))

function render() {
  const spec = buildGraphSpec()
  chart.setOption({
    backgroundColor: "#0f1730",
    tooltip: {
      formatter: (p) => {
        if (!p.data || !p.data.id) return ""
        const id = p.data.id
        if (id === "Result") return "Inference Result"
        const st = agentStates[id] || "-"
        return `${id}<br/>${st}`
      },
    },
    series: [
      {
        type: "graph",
        layout: "none",
        roam: true,
        label: { show: true, color: "#e7eaf3" },
        data: spec.nodes.map((n) => ({
          ...n,
          itemStyle: { color: nodeColor(n.id), borderColor: "#1a2240", borderWidth: 1 },
        })),
        links: spec.links.map((l) => ({
          ...l,
          lineStyle: { color: "#22305c", opacity: 0.35, width: 1 },
        })),
      },
    ],
  })
}

function updateProbePanel(id) {
  if (!id) return
  elId.textContent = id

  if (id === "Result") {
    elState.textContent = "-"
    elErr.textContent = "-"
    if (inferenceResult) {
      elWJson.textContent = "Final Probabilities:\n" + JSON.stringify(inferenceResult.probabilities, null, 2)
    } else {
      elWJson.textContent = "No result yet."
    }
    return
  }

  elState.textContent = agentStates[id] || "-"
  elErr.textContent = agentErrors[id] || "-"

  let content = ""
  if (id === "coordinator") {
    content = "Coordinator node (no trainable parameters)\n"
  } else if (agentWeights[id]) {
    const w = agentWeights[id].weights
    const b = agentWeights[id].bias
    content = `Bias:\n${b}\n\nWeights [len=${w.length}]:\n${JSON.stringify(w, null, 2)}\n\n`
  } else {
    content = "Weights not loaded yet.\n\n"
  }

  if (agentLLMRequests[id]) {
    content += `\n--- Latest LLM Request ---\n${JSON.stringify(agentLLMRequests[id], null, 2)}\n`
  }
  if (agentLLMResponses[id]) {
    content += `\n--- Latest LLM Response ---\n${JSON.stringify(agentLLMResponses[id], null, 2)}\n`
  }

  elWJson.textContent = content
}

chart.on("click", (params) => {
  const id = params?.data?.id
  updateProbePanel(id)
})

render()

function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws"
  return `${proto}://${window.location.host}/ws`
}

let ws = null

function connect() {
  ws = new WebSocket(wsUrl())
  ws.onopen = () => {}
  ws.onmessage = (ev) => {
    let msg
    try {
      msg = JSON.parse(ev.data)
    } catch {
      return
    }
    handleEvent(msg)
  }
  ws.onclose = () => {
    setTimeout(connect, 1000)
  }
}

function handleEvent(evt) {
  if (evt.type === "STATE_SNAPSHOT") {
    const agents = evt.data?.agents || {}
    for (const [id, info] of Object.entries(agents)) {
      agentStates[id] = info.state
      agentErrors[id] = info.last_error || "-"
    }
    render()
    return
  }

  if (evt.type === "AGENT_STATE_CHANGE") {
    const id = evt.data.agent_id
    agentStates[id] = evt.data.state
    if (evt.data.error) agentErrors[id] = evt.data.error
    if (elId.textContent === id) {
      elState.textContent = agentStates[id] || "-"
      elErr.textContent = agentErrors[id] || "-"
    }
    render()
    return
  }

  if (evt.type === "WEIGHT_LOADED") {
    const id = evt.data.agent_id
    agentWeights[id] = { weights: evt.data.weights, bias: evt.data.bias }
    // Visual feedback for loaded weights
    agentStates[id] = "UPDATED"
    agentErrors[id] = "-" // Clear previous error visually
    render()
    setTimeout(() => {
      // Don't fall back to ERROR if it was just loaded
      agentStates[id] = "WAITING_FORWARD_INPUT"
      render()
    }, 300)
    return
  }

  if (evt.type === "LLM_INTERACTION") {
    const { agent_id, step_id, op, phase, attempt, request, response, error } = evt.data
    
    if (phase === "request") {
      agentLLMRequests[agent_id] = request
      // Clear previous response/error when new request starts
      agentLLMResponses[agent_id] = null
    } else if (phase === "response") {
      agentLLMResponses[agent_id] = response
    } else if (phase === "error") {
      agentLLMResponses[agent_id] = { error }
    }

    if (elId.textContent === agent_id) {
      updateProbePanel(agent_id)
    }

    const el = document.createElement("div")
    el.className = `log-entry ${op}`
    if (phase === "request") {
      el.textContent = `[${agent_id}] ${op.toUpperCase()} @ ${step_id} (attempt ${attempt})\nReq: ${JSON.stringify(request)}`
    } else if (phase === "response") {
      el.textContent = `[${agent_id}] ${op.toUpperCase()} @ ${step_id} (attempt ${attempt})\nResp: ${JSON.stringify(response)}`
    } else {
      el.textContent = `[${agent_id}] ${op.toUpperCase()} @ ${step_id} (attempt ${attempt})\nErr: ${error}`
      el.classList.add("log-error")
    }
    elLogs.appendChild(el)
    elLogs.scrollTop = elLogs.scrollHeight
    return
  }

  if (evt.type === "INFERENCE_RESULT") {
    inferenceResult = evt.data
    render()
    if (elId.textContent === "Result") updateProbePanel("Result")
    alert(`Inference Done!\nPred Class: ${evt.data.prediction}\nConf: ${(evt.data.confidence*100).toFixed(2)}%`)
    return
  }

  if (evt.type === "GLOBAL_METRICS") {
    elStep.textContent = `${evt.data.sample_id}`
    elLoss.textContent = Number(evt.data.current_loss || 0).toFixed(6)
    elAcc.textContent = Number(evt.data.accuracy || 0).toFixed(2)
    elLP.textContent = `${evt.data.label}/${evt.data.pred}`
    return
  }

  if (evt.type === "ERROR") {
    elErr.textContent = evt.data.error || "error"
  }
}

function send(action) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return
  ws.send(JSON.stringify({ action }))
}

document.getElementById("btn-start").addEventListener("click", () => send("START_TRAINING"))
document.getElementById("btn-pause").addEventListener("click", () => send("PAUSE_TRAINING"))
document.getElementById("btn-reset").addEventListener("click", () => send("RESET"))

inpWeights.addEventListener("change", async (e) => {
  const file = e.target.files[0]
  if (!file) return
  const fd = new FormData()
  fd.append("file", file)
  try {
    const res = await fetch("/api/weights/load", { method: "POST", body: fd })
    const data = await res.json()
    if (!data.ok) alert("Load weights failed: " + data.error)
  } catch (err) {
    alert("Error: " + err)
  }
  e.target.value = ""
})

btnInfer.addEventListener("click", async () => {
  const f1 = document.getElementById("f1").value
  const f2 = document.getElementById("f2").value
  const f3 = document.getElementById("f3").value
  const f4 = document.getElementById("f4").value

  const features = [Number(f1), Number(f2), Number(f3), Number(f4)]
  
  inferenceResult = null
  render()
  if (elId.textContent === "Result") updateProbePanel("Result")

  try {
    const res = await fetch("/api/infer", { 
      method: "POST", 
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    })
    const data = await res.json()
    if (!data.ok) {
      alert("Inference failed to start: " + data.error)
    }
  } catch (err) {
    console.error("Error starting inference:", err)
  }
})

connect()
