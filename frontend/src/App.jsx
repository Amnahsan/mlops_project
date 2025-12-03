import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import WeatherCard from './components/WeatherCard'
import ClothingAvatar from './components/ClothingAvatar'

function App() {
  const [weather, setWeather] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        // Default to Oslo, Norway
        const response = await axios.post('http://localhost:8000/predict', {
          latitude: 59.91,
          longitude: 10.75
        })
        setWeather(response.data)
        setLoading(false)
      } catch (err) {
        console.error(err)
        setError('Failed to fetch weather data. Is the API running?')
        setLoading(false)
      }
    }

    fetchWeather()
  }, [])

  return (
    <div className="app-container">
      <header>
        <h1>✨ Weather & Fits 👗</h1>
        <p>Your daily dose of weather wisdom and style suggestions.</p>
      </header>

      <main>
        {loading && <div className="loading">🔮 Gazing into the crystal ball...</div>}

        {error && <div className="error">⚠️ {error}</div>}

        {weather && (
          <div className="dashboard">
            <WeatherCard
              temp={weather.predicted_max_temp}
              date={weather.input_date}
            />
            <ClothingAvatar suggestion={weather.clothing_suggestion} />
          </div>
        )}
      </main>

      <footer>
        <p>Powered by OpenMeteo & Random Forests 🌲</p>
      </footer>
    </div>
  )
}

export default App
