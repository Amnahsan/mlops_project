import React from 'react'

const WeatherCard = ({ temp, date }) => {
    const formattedDate = new Date(date).toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    })

    return (
        <div className="card weather-card">
            <h3>📅 Forecast for {formattedDate}</h3>
            <div className="weather-temp">
                {Math.round(temp)}°C
            </div>
            <p>Max Temperature</p>
        </div>
    )
}

export default WeatherCard
