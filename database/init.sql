-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    prediction INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    user_label INTEGER,
    probabilities JSON
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction ON predictions(prediction);