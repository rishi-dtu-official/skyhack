SELECT
    scheduled_arrival_station_code,
    COUNT(*) AS flights,
    AVG(transfer_ratio) AS avg_transfer_ratio,
    AVG(hot_transfer_ratio) AS avg_hot_transfer_ratio,
    AVG(load_factor) AS avg_load_factor,
    AVG(difficulty_probability_model) AS avg_difficulty_score
FROM flight_features
GROUP BY 1
HAVING flights >= 5
ORDER BY avg_transfer_ratio DESC;
