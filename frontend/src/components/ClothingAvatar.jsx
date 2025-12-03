import React from 'react'

const ClothingAvatar = ({ suggestion }) => {
    // Enhanced mapping
    const getEmoji = (text) => {
        if (!text) return "🤔";
        const lower = text.toLowerCase();
        if (lower.includes("shorts")) return "🏖️";
        if (lower.includes("jeans")) return "👖";
        if (lower.includes("coat")) return "🧥";
        if (lower.includes("thermal")) return "🥶";
        if (lower.includes("jacket")) return "🧥";
        return "👕";
    }

    return (
        <div className="card clothing-card">
            <h3>Fit Check</h3>
            <div className="clothing-emoji">
                {getEmoji(suggestion)}
            </div>
            <p className="clothing-suggestion">{suggestion || "Thinking..."}</p>
        </div>
    )
}

export default ClothingAvatar
