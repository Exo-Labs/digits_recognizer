import './App.css';
import React, { useRef, useState } from 'react';
import CanvasDraw from "react-canvas-draw";

const api_url = process.env.REACT_APP_API_URL;

const Paint = () => {
    const svgRef = useRef()
    const [data, setData] = useState([])
    const [img, setImg] = useState(0)
    const [resp, setResp] = useState(false)
    const [toggle, setToggle] = useState(true)

    const request = () => {
      setToggle(false)
      const imgData = data.split(',')[1]
      const url = api_url + '/predict'
      fetch(url, {
      method: 'POST',
      headers: {
        'Accept':'*/*'
      },
      body: JSON.stringify(imgData),
    })
    .then(response => response.json())
    .then(data => {
      console.log('Success:', data);
      setImg(data.prediction);
      setResp(true);
      setToggle(true)
  })
}

    const captureData = (e) => {
      const ctx = svgRef.current.ctx.drawing.canvas.getContext('2d')
      ctx.globalCompositeOperation = "darken";
      ctx.fillStyle = "#fff";
      ctx.fillRect(0,0,400,400);

      const canvas = svgRef.current.ctx.drawing.canvas.toDataURL()
      setData(canvas)
  }
    const undo = () => {
      setData([])
      setImg(0)
      setResp(false)
      
      const canvas = svgRef.current
      canvas.clear()
    }

    return (
      <div className="App" onMouseOver={captureData} onMouseOut={captureData}>
        <p>The model is deployed on a free heroku server, please give it a few seconds to wake up.</p>
          <CanvasDraw
            onChange={captureData}
            ref={svgRef}
            brushColor="black"
            hideGrid='true'
            brushRadius='20'
            backgroundColor='white'
            allowTransparency='false'
            fillStyle='rgba(0,0,0,0.5)'
            />
          {!resp && <div>Please draw one digit in range 0-9</div>}
          {toggle ? <button onClick={request}>Recognize</button> : <button>Processing</button>}
          <button onClick={undo}>Undo</button>
          {resp && <div>The digit is: {img}</div>}
      </div>
    );
  }
  
export default Paint;




