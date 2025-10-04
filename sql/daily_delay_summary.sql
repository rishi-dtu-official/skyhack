SELECT
    CAST(scheduled_departure_date_local AS DATE) AS departure_date,
    AVG(departure_delay_minutes) AS avg_departure_delay,
    AVG(CASE WHEN departure_delay_minutes > 0 THEN 1 ELSE 0 END) AS pct_late_departures,
    AVG(difficulty_probability_model) AS avg_difficulty_score,
    AVG(total_pax) AS avg_passenger_count
FROM flight_features
GROUP BY 1
ORDER BY 1;
