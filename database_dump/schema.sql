--
-- PostgreSQL database dump
--

-- Dumped from database version 14.18 (Homebrew)
-- Dumped by pg_dump version 14.18 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: alert_category; Type: TYPE; Schema: public; Owner: adam
--

CREATE TYPE public.alert_category AS ENUM (
    'assignment_risk',
    'earnings_proximity',
    'opportunity_csp',
    'opportunity_cc',
    'iv_spike',
    'xtrades_new',
    'margin_warning',
    'theta_decay',
    'expiration_reminder',
    'goal_progress',
    'report_ready'
);


ALTER TYPE public.alert_category OWNER TO adam;

--
-- Name: alert_channel; Type: TYPE; Schema: public; Owner: adam
--

CREATE TYPE public.alert_channel AS ENUM (
    'telegram',
    'discord',
    'email',
    'push',
    'in_app'
);


ALTER TYPE public.alert_channel OWNER TO adam;

--
-- Name: alert_priority; Type: TYPE; Schema: public; Owner: adam
--

CREATE TYPE public.alert_priority AS ENUM (
    'urgent',
    'important',
    'informational'
);


ALTER TYPE public.alert_priority OWNER TO adam;

--
-- Name: dependency_type; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.dependency_type AS ENUM (
    'imports',
    'calls',
    'data_flow',
    'inherits',
    'implements',
    'uses_service',
    'uses_database',
    'requires_config',
    'triggers',
    'composes'
);


ALTER TYPE public.dependency_type OWNER TO postgres;

--
-- Name: enhancement_priority; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.enhancement_priority AS ENUM (
    'p0_critical',
    'p1_high',
    'p2_medium',
    'p3_low',
    'p4_backlog'
);


ALTER TYPE public.enhancement_priority OWNER TO postgres;

--
-- Name: issue_severity; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.issue_severity AS ENUM (
    'critical',
    'high',
    'medium',
    'low',
    'info'
);


ALTER TYPE public.issue_severity OWNER TO postgres;

--
-- Name: source_file_type; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.source_file_type AS ENUM (
    'python',
    'typescript',
    'typescript_react',
    'javascript',
    'sql',
    'shell',
    'config',
    'other'
);


ALTER TYPE public.source_file_type OWNER TO postgres;

--
-- Name: spec_category; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.spec_category AS ENUM (
    'core',
    'agents_trading',
    'agents_analysis',
    'agents_sports',
    'agents_monitoring',
    'agents_research',
    'agents_management',
    'agents_code',
    'backend_services',
    'backend_routers',
    'frontend_pages',
    'frontend_components',
    'integrations',
    'database',
    'infrastructure'
);


ALTER TYPE public.spec_category OWNER TO postgres;

--
-- Name: calculate_beat_rate(character varying, integer); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.calculate_beat_rate(p_symbol character varying, p_lookback_quarters integer DEFAULT 8) RETURNS numeric
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_beat_rate DECIMAL(5,2);
BEGIN
    SELECT
        ROUND(100.0 * COUNT(*) FILTER (WHERE beat_miss = 'beat') / NULLIF(COUNT(*), 0), 2)
    INTO v_beat_rate
    FROM (
        SELECT beat_miss
        FROM earnings_history
        WHERE symbol = p_symbol
        AND beat_miss IS NOT NULL
        ORDER BY report_date DESC
        LIMIT p_lookback_quarters
    ) recent_quarters;

    RETURN COALESCE(v_beat_rate, 0);
END;
$$;


ALTER FUNCTION public.calculate_beat_rate(p_symbol character varying, p_lookback_quarters integer) OWNER TO adam;

--
-- Name: calculate_goal_progress(integer); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.calculate_goal_progress(p_goal_id integer) RETURNS TABLE(current_value numeric, progress_pct numeric, trades_count integer, winning_trades integer)
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_goal ava_user_goals%ROWTYPE;
    v_start_date DATE;
    v_end_date DATE;
    v_total_premium DECIMAL(15,2);
    v_trade_count INTEGER;
    v_win_count INTEGER;
BEGIN
    -- Get goal details
    SELECT * INTO v_goal FROM ava_user_goals WHERE id = p_goal_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Calculate date range based on period type
    CASE v_goal.period_type
        WHEN 'monthly' THEN
            v_start_date := date_trunc('month', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::DATE;
        WHEN 'weekly' THEN
            v_start_date := date_trunc('week', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('week', CURRENT_DATE) + INTERVAL '6 days')::DATE;
        WHEN 'annual' THEN
            v_start_date := date_trunc('year', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('year', CURRENT_DATE) + INTERVAL '1 year' - INTERVAL '1 day')::DATE;
        ELSE
            v_start_date := v_goal.start_date;
            v_end_date := COALESCE(v_goal.end_date, CURRENT_DATE);
    END CASE;

    -- Calculate from trade_journal if it exists (otherwise return 0)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trade_journal') THEN
        SELECT
            COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0),
            COUNT(*),
            COUNT(*) FILTER (WHERE pnl > 0)
        INTO v_total_premium, v_trade_count, v_win_count
        FROM trade_journal
        WHERE closed_date >= v_start_date
          AND closed_date <= v_end_date;
    ELSE
        v_total_premium := 0;
        v_trade_count := 0;
        v_win_count := 0;
    END IF;

    -- Return results
    current_value := v_total_premium;
    progress_pct := CASE WHEN v_goal.target_value > 0
                        THEN LEAST((v_total_premium / v_goal.target_value * 100), 200)
                        ELSE 0 END;
    trades_count := v_trade_count;
    winning_trades := v_win_count;

    RETURN NEXT;
END;
$$;


ALTER FUNCTION public.calculate_goal_progress(p_goal_id integer) OWNER TO adam;

--
-- Name: FUNCTION calculate_goal_progress(p_goal_id integer); Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON FUNCTION public.calculate_goal_progress(p_goal_id integer) IS 'Calculate goal progress from trade journal data';


--
-- Name: calculate_rolling_accuracy(character varying, integer); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.calculate_rolling_accuracy(p_sport character varying, p_days integer DEFAULT 30) RETURNS TABLE(total_predictions bigint, correct_predictions bigint, accuracy_rate numeric, avg_confidence numeric)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_predictions,
        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)::BIGINT as correct_predictions,
        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 4) as accuracy_rate,
        ROUND(AVG(predicted_probability), 4) as avg_confidence
    FROM prediction_results
    WHERE sport = p_sport
      AND game_completed_at IS NOT NULL
      AND game_completed_at >= NOW() - (p_days || ' days')::INTERVAL;
END;
$$;


ALTER FUNCTION public.calculate_rolling_accuracy(p_sport character varying, p_days integer) OWNER TO postgres;

--
-- Name: calculate_win_probability(integer, integer, character varying); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.calculate_win_probability(score_diff integer, quarter integer, time_remaining character varying) RETURNS numeric
    LANGUAGE plpgsql IMMUTABLE
    AS $$
DECLARE
    time_left_seconds INTEGER;
    time_parts TEXT[];
    minutes INTEGER;
    seconds INTEGER;
    base_prob DECIMAL(5,2);
    time_factor DECIMAL(5,2);
BEGIN
    -- Parse time remaining (MM:SS format)
    time_parts := string_to_array(time_remaining, ':');
    minutes := COALESCE(time_parts[1]::INTEGER, 0);
    seconds := COALESCE(time_parts[2]::INTEGER, 0);

    -- Calculate total seconds remaining (including future quarters)
    time_left_seconds := (4 - quarter) * 900 + (minutes * 60) + seconds;

    -- Base probability from score differential (simplified model)
    -- Each point is worth roughly 2% win probability
    base_prob := 50.0 + (score_diff * 2.0);

    -- Time factor: less time = more certainty
    -- At quarter 4, 2 minutes left, multiply certainty by ~1.5x
    time_factor := 1.0 + ((3600 - time_left_seconds)::DECIMAL / 3600.0) * 0.5;

    -- Adjust base probability by time factor
    base_prob := 50.0 + ((base_prob - 50.0) * time_factor);

    -- Clamp between 0.1% and 99.9%
    RETURN GREATEST(0.1, LEAST(99.9, base_prob));
END;
$$;


ALTER FUNCTION public.calculate_win_probability(score_diff integer, quarter integer, time_remaining character varying) OWNER TO postgres;

--
-- Name: FUNCTION calculate_win_probability(score_diff integer, quarter integer, time_remaining character varying); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.calculate_win_probability(score_diff integer, quarter integer, time_remaining character varying) IS 'Estimate win probability based on score and time remaining';


--
-- Name: check_and_increment_rate_limit(character varying, public.alert_channel, integer); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.check_and_increment_rate_limit(p_user_id character varying, p_channel public.alert_channel, p_max_per_hour integer DEFAULT 10) RETURNS boolean
    LANGUAGE plpgsql
    AS $$
DECLARE
    current_window TIMESTAMP WITH TIME ZONE;
    current_count INTEGER;
BEGIN
    -- Get current hour window
    current_window := date_trunc('hour', NOW());

    -- Try to get or create rate limit record
    INSERT INTO ava_alert_rate_limits (user_id, channel, window_start, max_alerts, alerts_sent)
    VALUES (p_user_id, p_channel, current_window, p_max_per_hour, 1)
    ON CONFLICT (user_id, channel, window_start)
    DO UPDATE SET alerts_sent = ava_alert_rate_limits.alerts_sent + 1
    RETURNING alerts_sent INTO current_count;

    -- Check if under limit
    RETURN current_count <= p_max_per_hour;
END;
$$;


ALTER FUNCTION public.check_and_increment_rate_limit(p_user_id character varying, p_channel public.alert_channel, p_max_per_hour integer) OWNER TO adam;

--
-- Name: FUNCTION check_and_increment_rate_limit(p_user_id character varying, p_channel public.alert_channel, p_max_per_hour integer); Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON FUNCTION public.check_and_increment_rate_limit(p_user_id character varying, p_channel public.alert_channel, p_max_per_hour integer) IS 'Check rate limit and increment counter, returns TRUE if allowed';


--
-- Name: generate_alert_fingerprint(public.alert_category, character varying, jsonb); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.generate_alert_fingerprint(p_category public.alert_category, p_symbol character varying, p_metadata jsonb) RETURNS character varying
    LANGUAGE plpgsql
    AS $$
DECLARE
    fingerprint_key TEXT;
BEGIN
    -- Build fingerprint based on category
    CASE p_category
        WHEN 'assignment_risk' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'strike', '') || ':' ||
                              COALESCE(p_metadata->>'expiration', '');
        WHEN 'opportunity_csp', 'opportunity_cc' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'strike', '') || ':' ||
                              date_trunc('hour', NOW())::TEXT;
        WHEN 'xtrades_new' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_metadata->>'trade_id', '');
        WHEN 'earnings_proximity' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'earnings_date', '');
        ELSE
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              date_trunc('hour', NOW())::TEXT;
    END CASE;

    -- Return MD5 hash (32 chars)
    RETURN MD5(fingerprint_key);
END;
$$;


ALTER FUNCTION public.generate_alert_fingerprint(p_category public.alert_category, p_symbol character varying, p_metadata jsonb) OWNER TO adam;

--
-- Name: FUNCTION generate_alert_fingerprint(p_category public.alert_category, p_symbol character varying, p_metadata jsonb); Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON FUNCTION public.generate_alert_fingerprint(p_category public.alert_category, p_symbol character varying, p_metadata jsonb) IS 'Generate unique fingerprint for alert deduplication';


--
-- Name: get_dependency_chain(integer, character varying, integer); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_dependency_chain(p_spec_id integer, p_direction character varying DEFAULT 'downstream'::character varying, p_max_depth integer DEFAULT 3) RETURNS TABLE(spec_id integer, feature_id character varying, feature_name character varying, dependency_type public.dependency_type, depth integer, path integer[])
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF p_direction = 'downstream' THEN
        RETURN QUERY
        WITH RECURSIVE dependency_chain AS (
            -- Base case: direct dependencies
            SELECT
                sd.target_spec_id as spec_id,
                sd.dependency_type,
                1 as depth,
                ARRAY[p_spec_id, sd.target_spec_id] as path
            FROM ava_spec_dependencies sd
            WHERE sd.source_spec_id = p_spec_id

            UNION ALL

            -- Recursive case
            SELECT
                sd.target_spec_id,
                sd.dependency_type,
                dc.depth + 1,
                dc.path || sd.target_spec_id
            FROM ava_spec_dependencies sd
            JOIN dependency_chain dc ON sd.source_spec_id = dc.spec_id
            WHERE dc.depth < p_max_depth
              AND NOT sd.target_spec_id = ANY(dc.path)
        )
        SELECT
            dc.spec_id,
            fs.feature_id,
            fs.feature_name,
            dc.dependency_type,
            dc.depth,
            dc.path
        FROM dependency_chain dc
        JOIN ava_feature_specs fs ON dc.spec_id = fs.id
        WHERE fs.is_current = TRUE
        ORDER BY dc.depth, fs.feature_name;
    ELSE
        RETURN QUERY
        WITH RECURSIVE dependency_chain AS (
            -- Base case: what I depend on
            SELECT
                sd.source_spec_id as spec_id,
                sd.dependency_type,
                1 as depth,
                ARRAY[p_spec_id, sd.source_spec_id] as path
            FROM ava_spec_dependencies sd
            WHERE sd.target_spec_id = p_spec_id

            UNION ALL

            -- Recursive case
            SELECT
                sd.source_spec_id,
                sd.dependency_type,
                dc.depth + 1,
                dc.path || sd.source_spec_id
            FROM ava_spec_dependencies sd
            JOIN dependency_chain dc ON sd.target_spec_id = dc.spec_id
            WHERE dc.depth < p_max_depth
              AND NOT sd.source_spec_id = ANY(dc.path)
        )
        SELECT
            dc.spec_id,
            fs.feature_id,
            fs.feature_name,
            dc.dependency_type,
            dc.depth,
            dc.path
        FROM dependency_chain dc
        JOIN ava_feature_specs fs ON dc.spec_id = fs.id
        WHERE fs.is_current = TRUE
        ORDER BY dc.depth, fs.feature_name;
    END IF;
END;
$$;


ALTER FUNCTION public.get_dependency_chain(p_spec_id integer, p_direction character varying, p_max_depth integer) OWNER TO postgres;

--
-- Name: FUNCTION get_dependency_chain(p_spec_id integer, p_direction character varying, p_max_depth integer); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.get_dependency_chain(p_spec_id integer, p_direction character varying, p_max_depth integer) IS 'Get upstream or downstream dependency chain with depth control';


--
-- Name: get_features_by_integration(character varying); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_features_by_integration(p_integration_name character varying) RETURNS TABLE(spec_id integer, feature_id character varying, feature_name character varying, category public.spec_category, integration_type character varying, is_critical boolean)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        si.integration_type,
        si.is_critical
    FROM ava_feature_specs fs
    JOIN ava_spec_integrations si ON fs.id = si.spec_id
    WHERE si.integration_name ILIKE '%' || p_integration_name || '%'
      AND fs.is_current = TRUE
      AND fs.status = 'active'
    ORDER BY si.is_critical DESC, fs.feature_name;
END;
$$;


ALTER FUNCTION public.get_features_by_integration(p_integration_name character varying) OWNER TO postgres;

--
-- Name: FUNCTION get_features_by_integration(p_integration_name character varying); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.get_features_by_integration(p_integration_name character varying) IS 'Find all features using a specific integration (e.g., Robinhood, Redis)';


--
-- Name: get_features_by_table(character varying); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_features_by_table(p_table_name character varying) RETURNS TABLE(spec_id integer, feature_id character varying, feature_name character varying, category public.spec_category, usage_type character varying, is_owner boolean)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        dt.usage_type,
        dt.is_owner
    FROM ava_feature_specs fs
    JOIN ava_spec_database_tables dt ON fs.id = dt.spec_id
    WHERE dt.table_name ILIKE '%' || p_table_name || '%'
      AND fs.is_current = TRUE
      AND fs.status = 'active'
    ORDER BY dt.is_owner DESC, fs.feature_name;
END;
$$;


ALTER FUNCTION public.get_features_by_table(p_table_name character varying) OWNER TO postgres;

--
-- Name: FUNCTION get_features_by_table(p_table_name character varying); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.get_features_by_table(p_table_name character varying) IS 'Find all features using a specific database table';


--
-- Name: get_low_efficiency_features(numeric, public.spec_category); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_low_efficiency_features(p_threshold numeric DEFAULT 7.0, p_category public.spec_category DEFAULT NULL::public.spec_category) RETURNS TABLE(spec_id integer, feature_id character varying, feature_name character varying, category public.spec_category, overall_rating numeric, priority_level character varying, weaknesses text[])
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        er.overall_rating,
        er.priority_level,
        er.weaknesses
    FROM ava_feature_specs fs
    JOIN ava_spec_efficiency_ratings er ON fs.id = er.spec_id
    WHERE fs.is_current = TRUE
      AND fs.status = 'active'
      AND (p_category IS NULL OR fs.category = p_category)
      AND er.overall_rating < p_threshold
    ORDER BY er.overall_rating ASC;
END;
$$;


ALTER FUNCTION public.get_low_efficiency_features(p_threshold numeric, p_category public.spec_category) OWNER TO postgres;

--
-- Name: FUNCTION get_low_efficiency_features(p_threshold numeric, p_category public.spec_category); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.get_low_efficiency_features(p_threshold numeric, p_category public.spec_category) IS 'Find features with efficiency below threshold';


--
-- Name: get_next_earnings_date(character varying); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.get_next_earnings_date(p_symbol character varying) RETURNS date
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_next_date DATE;
BEGIN
    SELECT earnings_date
    INTO v_next_date
    FROM earnings_events
    WHERE symbol = p_symbol
    AND has_occurred = FALSE
    AND earnings_date >= CURRENT_DATE
    ORDER BY earnings_date
    LIMIT 1;

    RETURN v_next_date;
END;
$$;


ALTER FUNCTION public.get_next_earnings_date(p_symbol character varying) OWNER TO adam;

--
-- Name: get_recommendation_accuracy(character varying, integer); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.get_recommendation_accuracy(p_user_id character varying DEFAULT 'default_user'::character varying, p_days integer DEFAULT 30) RETURNS TABLE(total_recommendations integer, recommendations_with_outcome integer, correct_recommendations integer, accuracy_pct numeric, avg_confidence numeric, avg_pnl numeric)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_recommendations,
        COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL)::INTEGER as recommendations_with_outcome,
        COUNT(*) FILTER (WHERE recommendation_correct = TRUE)::INTEGER as correct_recommendations,
        CASE
            WHEN COUNT(*) FILTER (WHERE recommendation_correct IS NOT NULL) > 0
            THEN (COUNT(*) FILTER (WHERE recommendation_correct = TRUE)::DECIMAL /
                  COUNT(*) FILTER (WHERE recommendation_correct IS NOT NULL) * 100)
            ELSE 0
        END as accuracy_pct,
        AVG(confidence_score)::DECIMAL(3,2) as avg_confidence,
        AVG(actual_pnl)::DECIMAL(15,2) as avg_pnl
    FROM ava_chat_recommendations
    WHERE user_id = p_user_id
      AND created_at >= NOW() - (p_days || ' days')::INTERVAL;
END;
$$;


ALTER FUNCTION public.get_recommendation_accuracy(p_user_id character varying, p_days integer) OWNER TO adam;

--
-- Name: FUNCTION get_recommendation_accuracy(p_user_id character varying, p_days integer); Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON FUNCTION public.get_recommendation_accuracy(p_user_id character varying, p_days integer) IS 'Get recommendation accuracy statistics for a user';


--
-- Name: search_specs_by_embedding(double precision[], integer, numeric, public.spec_category); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.search_specs_by_embedding(p_query_embedding double precision[], p_limit integer DEFAULT 10, p_similarity_threshold numeric DEFAULT 0.7, p_category public.spec_category DEFAULT NULL::public.spec_category) RETURNS TABLE(spec_id integer, feature_id character varying, feature_name character varying, category public.spec_category, purpose text, similarity numeric)
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- With FLOAT[], we use a simpler approach (fallback until pgvector available)
    -- This returns all matching features ordered by name (proper similarity requires pgvector)
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        fs.purpose,
        1.0::DECIMAL as similarity  -- Placeholder until pgvector
    FROM ava_feature_specs fs
    WHERE fs.is_current = TRUE
      AND fs.status = 'active'
      AND (p_category IS NULL OR fs.category = p_category)
    ORDER BY fs.feature_name
    LIMIT p_limit;
END;
$$;


ALTER FUNCTION public.search_specs_by_embedding(p_query_embedding double precision[], p_limit integer, p_similarity_threshold numeric, p_category public.spec_category) OWNER TO postgres;

--
-- Name: FUNCTION search_specs_by_embedding(p_query_embedding double precision[], p_limit integer, p_similarity_threshold numeric, p_category public.spec_category); Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON FUNCTION public.search_specs_by_embedding(p_query_embedding double precision[], p_limit integer, p_similarity_threshold numeric, p_category public.spec_category) IS 'Semantic search for feature specs (requires pgvector for true similarity)';


--
-- Name: update_ava_advisor_updated_at(); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.update_ava_advisor_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_ava_advisor_updated_at() OWNER TO adam;

--
-- Name: update_last_updated_column(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_last_updated_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_last_updated_column() OWNER TO postgres;

--
-- Name: update_last_updated_timestamp(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_last_updated_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_last_updated_timestamp() OWNER TO postgres;

--
-- Name: update_modified_column(); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.update_modified_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_modified_column() OWNER TO adam;

--
-- Name: update_scanner_watchlists_timestamp(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_scanner_watchlists_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_scanner_watchlists_timestamp() OWNER TO postgres;

--
-- Name: update_spec_updated_at(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_spec_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_spec_updated_at() OWNER TO postgres;

--
-- Name: update_sync_status(character varying, character varying, integer, integer, text); Type: FUNCTION; Schema: public; Owner: adam
--

CREATE FUNCTION public.update_sync_status(p_symbol character varying, p_status character varying, p_historical_count integer DEFAULT 0, p_upcoming_count integer DEFAULT 0, p_error_message text DEFAULT NULL::text) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO earnings_sync_status (
        symbol,
        last_sync_at,
        last_sync_status,
        last_error_message,
        historical_quarters_found,
        upcoming_events_found,
        total_syncs,
        failed_syncs,
        next_sync_at,
        updated_at
    ) VALUES (
        p_symbol,
        NOW(),
        p_status,
        p_error_message,
        p_historical_count,
        p_upcoming_count,
        1,
        CASE WHEN p_status = 'failed' THEN 1 ELSE 0 END,
        NOW() + INTERVAL '24 hours',
        NOW()
    )
    ON CONFLICT (symbol) DO UPDATE SET
        last_sync_at = NOW(),
        last_sync_status = p_status,
        last_error_message = p_error_message,
        historical_quarters_found = p_historical_count,
        upcoming_events_found = p_upcoming_count,
        total_syncs = earnings_sync_status.total_syncs + 1,
        failed_syncs = earnings_sync_status.failed_syncs +
            CASE WHEN p_status = 'failed' THEN 1 ELSE 0 END,
        next_sync_at = NOW() + INTERVAL '24 hours',
        updated_at = NOW();
END;
$$;


ALTER FUNCTION public.update_sync_status(p_symbol character varying, p_status character varying, p_historical_count integer, p_upcoming_count integer, p_error_message text) OWNER TO adam;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: agent_execution_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.agent_execution_log (
    id integer NOT NULL,
    agent_name character varying(100) NOT NULL,
    execution_id character varying(100) NOT NULL,
    input_text text,
    result jsonb,
    error text,
    response_time_ms double precision,
    user_id character varying(100),
    platform character varying(50),
    "timestamp" timestamp without time zone DEFAULT now()
);


ALTER TABLE public.agent_execution_log OWNER TO postgres;

--
-- Name: agent_execution_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.agent_execution_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.agent_execution_log_id_seq OWNER TO postgres;

--
-- Name: agent_execution_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.agent_execution_log_id_seq OWNED BY public.agent_execution_log.id;


--
-- Name: agent_feedback; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.agent_feedback (
    id integer NOT NULL,
    agent_name character varying(100) NOT NULL,
    feedback_type character varying(50) NOT NULL,
    feedback_text text,
    user_id character varying(100),
    "timestamp" timestamp without time zone DEFAULT now(),
    resolved boolean DEFAULT false
);


ALTER TABLE public.agent_feedback OWNER TO postgres;

--
-- Name: agent_feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.agent_feedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.agent_feedback_id_seq OWNER TO postgres;

--
-- Name: agent_feedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.agent_feedback_id_seq OWNED BY public.agent_feedback.id;


--
-- Name: agent_memory; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.agent_memory (
    id integer NOT NULL,
    agent_name character varying(100) NOT NULL,
    memory_key character varying(255) NOT NULL,
    memory_value jsonb,
    context jsonb,
    created_at timestamp without time zone DEFAULT now(),
    last_accessed timestamp without time zone DEFAULT now(),
    access_count integer DEFAULT 0
);


ALTER TABLE public.agent_memory OWNER TO postgres;

--
-- Name: agent_memory_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.agent_memory_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.agent_memory_id_seq OWNER TO postgres;

--
-- Name: agent_memory_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.agent_memory_id_seq OWNED BY public.agent_memory.id;


--
-- Name: agent_performance; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.agent_performance (
    agent_name character varying(100) NOT NULL,
    total_executions integer DEFAULT 0,
    successful_executions integer DEFAULT 0,
    failed_executions integer DEFAULT 0,
    average_response_time double precision DEFAULT 0.0,
    last_execution timestamp without time zone,
    success_rate double precision DEFAULT 0.0,
    user_satisfaction double precision DEFAULT 0.0,
    last_updated timestamp without time zone DEFAULT now()
);


ALTER TABLE public.agent_performance OWNER TO postgres;

--
-- Name: ai_betting_recommendations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ai_betting_recommendations (
    id integer NOT NULL,
    sport character varying(20) NOT NULL,
    game_id character varying(50) NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    bet_type character varying(30) NOT NULL,
    pick character varying(100) NOT NULL,
    odds integer,
    confidence integer,
    win_probability numeric(5,2),
    expected_value numeric(6,2),
    key_factors jsonb,
    reasoning text,
    game_state jsonb,
    odds_movement_factor numeric(4,2),
    is_settled boolean DEFAULT false,
    result character varying(10),
    settled_at timestamp with time zone,
    CONSTRAINT ai_betting_recommendations_confidence_check CHECK (((confidence >= 0) AND (confidence <= 100))),
    CONSTRAINT chk_bet_type CHECK (((bet_type)::text = ANY ((ARRAY['spread'::character varying, 'moneyline'::character varying, 'over'::character varying, 'under'::character varying, 'live_spread'::character varying, 'live_total'::character varying, 'live_ml'::character varying])::text[])))
);


ALTER TABLE public.ai_betting_recommendations OWNER TO postgres;

--
-- Name: ai_betting_recommendations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ai_betting_recommendations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ai_betting_recommendations_id_seq OWNER TO postgres;

--
-- Name: ai_betting_recommendations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ai_betting_recommendations_id_seq OWNED BY public.ai_betting_recommendations.id;


--
-- Name: automation_executions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.automation_executions (
    id bigint NOT NULL,
    automation_id integer NOT NULL,
    celery_task_id character varying(255),
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    duration_seconds numeric(10,3),
    status character varying(50) DEFAULT 'running'::character varying NOT NULL,
    result jsonb,
    error_message text,
    error_traceback text,
    records_processed integer,
    triggered_by character varying(100) DEFAULT 'scheduler'::character varying,
    worker_hostname character varying(255),
    CONSTRAINT chk_execution_status CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'running'::character varying, 'success'::character varying, 'failed'::character varying, 'revoked'::character varying, 'timeout'::character varying, 'skipped'::character varying])::text[])))
);


ALTER TABLE public.automation_executions OWNER TO postgres;

--
-- Name: TABLE automation_executions; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.automation_executions IS 'Execution history and performance tracking for automations';


--
-- Name: automation_executions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.automation_executions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.automation_executions_id_seq OWNER TO postgres;

--
-- Name: automation_executions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.automation_executions_id_seq OWNED BY public.automation_executions.id;


--
-- Name: automation_state_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.automation_state_log (
    id integer NOT NULL,
    automation_id integer NOT NULL,
    previous_state boolean,
    new_state boolean NOT NULL,
    changed_at timestamp with time zone DEFAULT now(),
    changed_by character varying(100),
    reason text,
    affected_task_ids text[]
);


ALTER TABLE public.automation_state_log OWNER TO postgres;

--
-- Name: TABLE automation_state_log; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.automation_state_log IS 'Audit trail for enable/disable actions';


--
-- Name: automation_state_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.automation_state_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.automation_state_log_id_seq OWNER TO postgres;

--
-- Name: automation_state_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.automation_state_log_id_seq OWNED BY public.automation_state_log.id;


--
-- Name: automations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.automations (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    display_name character varying(200) NOT NULL,
    automation_type character varying(50) NOT NULL,
    celery_task_name character varying(255),
    schedule_type character varying(50),
    schedule_config jsonb,
    schedule_display character varying(100),
    queue character varying(100) DEFAULT 'default'::character varying,
    category character varying(100) NOT NULL,
    description text,
    is_enabled boolean DEFAULT true NOT NULL,
    enabled_updated_at timestamp with time zone,
    enabled_updated_by character varying(100),
    timeout_seconds integer DEFAULT 300,
    max_retries integer DEFAULT 3,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_automation_type CHECK (((automation_type)::text = ANY ((ARRAY['celery_beat'::character varying, 'celery_task'::character varying])::text[])))
);


ALTER TABLE public.automations OWNER TO postgres;

--
-- Name: TABLE automations; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.automations IS 'Master registry of all automated tasks managed by the Developer Console';


--
-- Name: automations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.automations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.automations_id_seq OWNER TO postgres;

--
-- Name: automations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.automations_id_seq OWNED BY public.automations.id;


--
-- Name: ava_alert_deliveries; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_alert_deliveries (
    id integer NOT NULL,
    alert_id integer,
    channel public.alert_channel NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying,
    sent_at timestamp with time zone,
    error_message text,
    retry_count integer DEFAULT 0,
    external_message_id character varying(100),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_alert_deliveries OWNER TO adam;

--
-- Name: TABLE ava_alert_deliveries; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_alert_deliveries IS 'Tracks alert delivery status per channel with retry support';


--
-- Name: ava_alert_deliveries_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_alert_deliveries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_alert_deliveries_id_seq OWNER TO adam;

--
-- Name: ava_alert_deliveries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_alert_deliveries_id_seq OWNED BY public.ava_alert_deliveries.id;


--
-- Name: ava_alert_preferences; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_alert_preferences (
    id integer NOT NULL,
    user_id character varying(100) DEFAULT 'default_user'::character varying NOT NULL,
    platform character varying(50) DEFAULT 'web'::character varying NOT NULL,
    category public.alert_category NOT NULL,
    enabled boolean DEFAULT true,
    priority_threshold public.alert_priority DEFAULT 'informational'::public.alert_priority,
    channels text[] DEFAULT ARRAY['telegram'::text],
    quiet_hours_enabled boolean DEFAULT false,
    quiet_hours_start time without time zone,
    quiet_hours_end time without time zone,
    max_per_hour integer DEFAULT 10,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_alert_preferences OWNER TO adam;

--
-- Name: TABLE ava_alert_preferences; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_alert_preferences IS 'User preferences for alert categories and delivery channels';


--
-- Name: ava_alert_preferences_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_alert_preferences_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_alert_preferences_id_seq OWNER TO adam;

--
-- Name: ava_alert_preferences_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_alert_preferences_id_seq OWNED BY public.ava_alert_preferences.id;


--
-- Name: ava_alert_rate_limits; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_alert_rate_limits (
    id integer NOT NULL,
    user_id character varying(100) DEFAULT 'default_user'::character varying NOT NULL,
    channel public.alert_channel NOT NULL,
    window_start timestamp with time zone NOT NULL,
    window_duration_minutes integer DEFAULT 60,
    max_alerts integer DEFAULT 10,
    alerts_sent integer DEFAULT 0
);


ALTER TABLE public.ava_alert_rate_limits OWNER TO adam;

--
-- Name: TABLE ava_alert_rate_limits; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_alert_rate_limits IS 'Rate limiting state to prevent alert spam';


--
-- Name: ava_alert_rate_limits_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_alert_rate_limits_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_alert_rate_limits_id_seq OWNER TO adam;

--
-- Name: ava_alert_rate_limits_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_alert_rate_limits_id_seq OWNED BY public.ava_alert_rate_limits.id;


--
-- Name: ava_alerts; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_alerts (
    id integer NOT NULL,
    category public.alert_category NOT NULL,
    priority public.alert_priority NOT NULL,
    title character varying(255) NOT NULL,
    message text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    symbol character varying(20),
    position_id uuid,
    fingerprint character varying(64),
    is_active boolean DEFAULT true,
    is_read boolean DEFAULT false,
    read_at timestamp with time zone,
    expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_alerts OWNER TO adam;

--
-- Name: TABLE ava_alerts; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_alerts IS 'Proactive alerts for position risks, opportunities, and reports';


--
-- Name: COLUMN ava_alerts.metadata; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_alerts.metadata IS 'Flexible JSON for alert-specific data (strike, expiry, score, etc.)';


--
-- Name: COLUMN ava_alerts.fingerprint; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_alerts.fingerprint IS 'SHA256 hash for deduplication (prevents duplicate alerts)';


--
-- Name: ava_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_alerts_id_seq OWNER TO adam;

--
-- Name: ava_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_alerts_id_seq OWNED BY public.ava_alerts.id;


--
-- Name: ava_chat_recommendations; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_chat_recommendations (
    id integer NOT NULL,
    user_id character varying(100) DEFAULT 'default_user'::character varying NOT NULL,
    platform character varying(50) DEFAULT 'web'::character varying NOT NULL,
    conversation_id character varying(100),
    recommendation_type character varying(50) NOT NULL,
    symbol character varying(20),
    strategy character varying(100),
    recommendation_text text NOT NULL,
    confidence_score numeric(3,2),
    reasoning text,
    context_snapshot jsonb,
    rag_sources_used text[],
    agents_used text[],
    user_action character varying(50),
    user_action_at timestamp with time zone,
    trade_id uuid,
    actual_outcome character varying(50),
    actual_pnl numeric(15,2),
    outcome_recorded_at timestamp with time zone,
    recommendation_correct boolean,
    feedback_score integer,
    feedback_text text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_chat_recommendations OWNER TO adam;

--
-- Name: TABLE ava_chat_recommendations; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_chat_recommendations IS 'Tracks chatbot recommendations with outcomes for learning';


--
-- Name: COLUMN ava_chat_recommendations.context_snapshot; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_chat_recommendations.context_snapshot IS 'JSON snapshot of portfolio and market state when recommendation was made';


--
-- Name: COLUMN ava_chat_recommendations.recommendation_correct; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_chat_recommendations.recommendation_correct IS 'Whether the recommendation was ultimately correct (for learning)';


--
-- Name: ava_chat_recommendations_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_chat_recommendations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_chat_recommendations_id_seq OWNER TO adam;

--
-- Name: ava_chat_recommendations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_chat_recommendations_id_seq OWNED BY public.ava_chat_recommendations.id;


--
-- Name: ava_feature_specs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_feature_specs (
    id integer NOT NULL,
    feature_id character varying(100) NOT NULL,
    feature_name character varying(255) NOT NULL,
    category public.spec_category NOT NULL,
    subcategory character varying(100),
    purpose text NOT NULL,
    description text,
    key_responsibilities text[],
    version character varying(20) DEFAULT '1.0.0'::character varying,
    is_current boolean DEFAULT true,
    status character varying(50) DEFAULT 'active'::character varying,
    maturity_level character varying(50) DEFAULT 'stable'::character varying,
    technical_details jsonb DEFAULT '{}'::jsonb,
    embedding double precision[],
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    analyzed_at timestamp with time zone,
    CONSTRAINT chk_feature_status CHECK (((status)::text = ANY ((ARRAY['active'::character varying, 'deprecated'::character varying, 'planned'::character varying, 'experimental'::character varying, 'removed'::character varying])::text[])))
);


ALTER TABLE public.ava_feature_specs OWNER TO postgres;

--
-- Name: TABLE ava_feature_specs; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_feature_specs IS 'Master table for all AVA platform feature specifications';


--
-- Name: COLUMN ava_feature_specs.feature_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ava_feature_specs.feature_id IS 'Unique identifier matching code naming conventions';


--
-- Name: COLUMN ava_feature_specs.embedding; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ava_feature_specs.embedding IS 'Vector embedding for semantic similarity search (1536 dimensions)';


--
-- Name: ava_feature_specs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_feature_specs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_feature_specs_id_seq OWNER TO postgres;

--
-- Name: ava_feature_specs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_feature_specs_id_seq OWNED BY public.ava_feature_specs.id;


--
-- Name: ava_generated_reports; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_generated_reports (
    id integer NOT NULL,
    report_type character varying(50) NOT NULL,
    report_date date NOT NULL,
    title character varying(255) NOT NULL,
    content text NOT NULL,
    summary text,
    metrics jsonb DEFAULT '{}'::jsonb,
    telegram_sent boolean DEFAULT false,
    email_sent boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_generated_reports OWNER TO adam;

--
-- Name: TABLE ava_generated_reports; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_generated_reports IS 'Storage for generated reports (morning briefings, summaries, etc.)';


--
-- Name: ava_generated_reports_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_generated_reports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_generated_reports_id_seq OWNER TO adam;

--
-- Name: ava_generated_reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_generated_reports_id_seq OWNED BY public.ava_generated_reports.id;


--
-- Name: ava_goal_progress_history; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_goal_progress_history (
    id integer NOT NULL,
    goal_id integer,
    snapshot_date date NOT NULL,
    period_value numeric(15,2) NOT NULL,
    cumulative_value numeric(15,2) NOT NULL,
    progress_pct numeric(5,2) NOT NULL,
    trades_count integer DEFAULT 0,
    winning_trades integer DEFAULT 0,
    premium_collected numeric(15,2) DEFAULT 0,
    total_pnl numeric(15,2) DEFAULT 0,
    notes text,
    ai_analysis text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_goal_progress_history OWNER TO adam;

--
-- Name: TABLE ava_goal_progress_history; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_goal_progress_history IS 'Historical snapshots of goal progress for trend analysis';


--
-- Name: ava_goal_progress_history_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_goal_progress_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_goal_progress_history_id_seq OWNER TO adam;

--
-- Name: ava_goal_progress_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_goal_progress_history_id_seq OWNED BY public.ava_goal_progress_history.id;


--
-- Name: ava_iv_history; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_iv_history (
    id integer NOT NULL,
    symbol character varying(20) NOT NULL,
    date date NOT NULL,
    iv_30 numeric(6,2),
    iv_60 numeric(6,2),
    iv_rank numeric(5,2),
    iv_percentile numeric(5,2),
    stock_price numeric(10,2),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_iv_history OWNER TO adam;

--
-- Name: TABLE ava_iv_history; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_iv_history IS 'Historical IV data for spike detection and trend analysis';


--
-- Name: ava_iv_history_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_iv_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_iv_history_id_seq OWNER TO adam;

--
-- Name: ava_iv_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_iv_history_id_seq OWNED BY public.ava_iv_history.id;


--
-- Name: ava_learning_patterns; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_learning_patterns (
    id integer NOT NULL,
    user_id character varying(100) DEFAULT 'default_user'::character varying NOT NULL,
    platform character varying(50) DEFAULT 'web'::character varying NOT NULL,
    pattern_type character varying(50) NOT NULL,
    pattern_name character varying(200) NOT NULL,
    pattern_description text,
    pattern_conditions jsonb NOT NULL,
    sample_trades jsonb,
    sample_count integer DEFAULT 0,
    win_rate numeric(5,2),
    avg_pnl numeric(15,2),
    total_pnl numeric(15,2),
    avg_holding_days numeric(5,1),
    confidence_score numeric(3,2) DEFAULT 0.50,
    last_validated_at timestamp with time zone,
    validation_count integer DEFAULT 0,
    active boolean DEFAULT true,
    weight_multiplier numeric(3,2) DEFAULT 1.0,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_learning_patterns OWNER TO adam;

--
-- Name: TABLE ava_learning_patterns; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_learning_patterns IS 'Learned patterns from user trading history for personalization';


--
-- Name: COLUMN ava_learning_patterns.pattern_conditions; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_learning_patterns.pattern_conditions IS 'JSON conditions that define this pattern (IV, DTE, delta, etc.)';


--
-- Name: COLUMN ava_learning_patterns.weight_multiplier; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_learning_patterns.weight_multiplier IS 'Multiplier applied to recommendations matching this pattern';


--
-- Name: ava_learning_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_learning_patterns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_learning_patterns_id_seq OWNER TO adam;

--
-- Name: ava_learning_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_learning_patterns_id_seq OWNED BY public.ava_learning_patterns.id;


--
-- Name: ava_opportunity_scans; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_opportunity_scans (
    id integer NOT NULL,
    scan_type character varying(50) NOT NULL,
    watchlist_used character varying(100),
    symbols_scanned integer DEFAULT 0,
    opportunities_found integer DEFAULT 0,
    alerts_generated integer DEFAULT 0,
    top_opportunities jsonb DEFAULT '[]'::jsonb,
    scan_duration_ms integer,
    errors text[],
    started_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone
);


ALTER TABLE public.ava_opportunity_scans OWNER TO adam;

--
-- Name: TABLE ava_opportunity_scans; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_opportunity_scans IS 'History of automatic opportunity scans';


--
-- Name: ava_opportunity_scans_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_opportunity_scans_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_opportunity_scans_id_seq OWNER TO adam;

--
-- Name: ava_opportunity_scans_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_opportunity_scans_id_seq OWNED BY public.ava_opportunity_scans.id;


--
-- Name: ava_spec_api_endpoints; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_api_endpoints (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    method character varying(10) NOT NULL,
    path character varying(500) NOT NULL,
    router_name character varying(100),
    summary character varying(500),
    description text,
    request_model character varying(200),
    response_model character varying(200),
    query_params jsonb DEFAULT '[]'::jsonb,
    path_params jsonb DEFAULT '[]'::jsonb,
    requires_auth boolean DEFAULT true,
    required_permissions text[],
    is_active boolean DEFAULT true,
    deprecated_at timestamp with time zone,
    avg_response_time_ms integer,
    p95_response_time_ms integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_api_endpoints OWNER TO postgres;

--
-- Name: TABLE ava_spec_api_endpoints; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_api_endpoints IS 'API endpoints exposed by each feature';


--
-- Name: ava_spec_api_endpoints_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_api_endpoints_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_api_endpoints_id_seq OWNER TO postgres;

--
-- Name: ava_spec_api_endpoints_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_api_endpoints_id_seq OWNED BY public.ava_spec_api_endpoints.id;


--
-- Name: ava_spec_database_tables; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_database_tables (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    table_name character varying(200) NOT NULL,
    schema_name character varying(100) DEFAULT 'public'::character varying,
    usage_type character varying(50) NOT NULL,
    is_owner boolean DEFAULT false,
    columns_used text[],
    access_patterns text[],
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_database_tables OWNER TO postgres;

--
-- Name: TABLE ava_spec_database_tables; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_database_tables IS 'Database tables used by each feature';


--
-- Name: ava_spec_database_tables_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_database_tables_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_database_tables_id_seq OWNER TO postgres;

--
-- Name: ava_spec_database_tables_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_database_tables_id_seq OWNED BY public.ava_spec_database_tables.id;


--
-- Name: ava_spec_dependencies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_dependencies (
    id integer NOT NULL,
    source_spec_id integer NOT NULL,
    target_spec_id integer NOT NULL,
    dependency_type public.dependency_type NOT NULL,
    description text,
    is_critical boolean DEFAULT false,
    strength integer DEFAULT 5,
    bidirectional boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_no_self_dependency CHECK ((source_spec_id <> target_spec_id)),
    CONSTRAINT chk_strength CHECK (((strength >= 1) AND (strength <= 10)))
);


ALTER TABLE public.ava_spec_dependencies OWNER TO postgres;

--
-- Name: TABLE ava_spec_dependencies; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_dependencies IS 'Dependency graph between features for impact analysis';


--
-- Name: ava_spec_dependencies_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_dependencies_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_dependencies_id_seq OWNER TO postgres;

--
-- Name: ava_spec_dependencies_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_dependencies_id_seq OWNED BY public.ava_spec_dependencies.id;


--
-- Name: ava_spec_efficiency_ratings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_efficiency_ratings (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    overall_rating numeric(3,1) NOT NULL,
    code_completeness numeric(3,1),
    test_coverage numeric(3,1),
    performance numeric(3,1),
    error_handling numeric(3,1),
    documentation numeric(3,1),
    maintainability numeric(3,1),
    dependencies numeric(3,1),
    priority_level character varying(20),
    metrics jsonb DEFAULT '{}'::jsonb,
    analysis_summary text,
    strengths text[],
    weaknesses text[],
    quick_wins text[],
    assessed_by character varying(100) DEFAULT 'ai'::character varying,
    assessment_version character varying(20),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_rating_range CHECK (((overall_rating >= 1.0) AND (overall_rating <= 10.0) AND ((code_completeness IS NULL) OR ((code_completeness >= 1.0) AND (code_completeness <= 10.0))) AND ((test_coverage IS NULL) OR ((test_coverage >= 1.0) AND (test_coverage <= 10.0))) AND ((performance IS NULL) OR ((performance >= 1.0) AND (performance <= 10.0))) AND ((error_handling IS NULL) OR ((error_handling >= 1.0) AND (error_handling <= 10.0))) AND ((documentation IS NULL) OR ((documentation >= 1.0) AND (documentation <= 10.0))) AND ((maintainability IS NULL) OR ((maintainability >= 1.0) AND (maintainability <= 10.0))) AND ((dependencies IS NULL) OR ((dependencies >= 1.0) AND (dependencies <= 10.0)))))
);


ALTER TABLE public.ava_spec_efficiency_ratings OWNER TO postgres;

--
-- Name: TABLE ava_spec_efficiency_ratings; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_efficiency_ratings IS 'Efficiency ratings and assessments for each feature (7 dimensions)';


--
-- Name: ava_spec_efficiency_ratings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_efficiency_ratings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_efficiency_ratings_id_seq OWNER TO postgres;

--
-- Name: ava_spec_efficiency_ratings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_efficiency_ratings_id_seq OWNED BY public.ava_spec_efficiency_ratings.id;


--
-- Name: ava_spec_enhancements; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_enhancements (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    enhancement_title character varying(500) NOT NULL,
    enhancement_description text NOT NULL,
    priority public.enhancement_priority NOT NULL,
    estimated_effort character varying(50),
    expected_impact character varying(50),
    affected_areas text[],
    implementation_notes text,
    prerequisite_enhancements integer[],
    status character varying(50) DEFAULT 'proposed'::character varying,
    completed_at timestamp with time zone,
    ai_confidence numeric(3,2),
    reasoning text,
    proposed_at timestamp with time zone DEFAULT now(),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_enhancements OWNER TO postgres;

--
-- Name: TABLE ava_spec_enhancements; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_enhancements IS 'Enhancement opportunities identified for each feature';


--
-- Name: ava_spec_enhancements_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_enhancements_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_enhancements_id_seq OWNER TO postgres;

--
-- Name: ava_spec_enhancements_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_enhancements_id_seq OWNED BY public.ava_spec_enhancements.id;


--
-- Name: ava_spec_error_handling; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_error_handling (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    error_type character varying(200) NOT NULL,
    error_code character varying(50),
    handling_strategy text NOT NULL,
    user_message_template text,
    is_recoverable boolean DEFAULT true,
    retry_enabled boolean DEFAULT false,
    max_retries integer,
    log_level character varying(20) DEFAULT 'ERROR'::character varying,
    alert_threshold integer,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_error_handling OWNER TO postgres;

--
-- Name: TABLE ava_spec_error_handling; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_error_handling IS 'Error handling specifications for each feature';


--
-- Name: ava_spec_error_handling_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_error_handling_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_error_handling_id_seq OWNER TO postgres;

--
-- Name: ava_spec_error_handling_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_error_handling_id_seq OWNED BY public.ava_spec_error_handling.id;


--
-- Name: ava_spec_integrations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_integrations (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    integration_name character varying(200) NOT NULL,
    integration_type character varying(100) NOT NULL,
    config_requirements jsonb DEFAULT '[]'::jsonb,
    env_variables text[],
    endpoint_pattern character varying(500),
    auth_type character varying(100),
    retry_strategy character varying(200),
    fallback_behavior text,
    circuit_breaker_enabled boolean DEFAULT false,
    health_check_endpoint character varying(500),
    is_critical boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_integrations OWNER TO postgres;

--
-- Name: TABLE ava_spec_integrations; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_integrations IS 'External integration points for each feature';


--
-- Name: ava_spec_integrations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_integrations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_integrations_id_seq OWNER TO postgres;

--
-- Name: ava_spec_integrations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_integrations_id_seq OWNED BY public.ava_spec_integrations.id;


--
-- Name: ava_spec_known_issues; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_known_issues (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    issue_title character varying(500) NOT NULL,
    issue_description text NOT NULL,
    severity public.issue_severity NOT NULL,
    issue_type character varying(100),
    affected_files text[],
    status character varying(50) DEFAULT 'open'::character varying,
    resolution text,
    resolved_at timestamp with time zone,
    github_issue_url character varying(500),
    related_commits text[],
    reported_at timestamp with time zone DEFAULT now(),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_known_issues OWNER TO postgres;

--
-- Name: TABLE ava_spec_known_issues; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_known_issues IS 'Known issues and bugs for each feature';


--
-- Name: ava_spec_known_issues_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_known_issues_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_known_issues_id_seq OWNER TO postgres;

--
-- Name: ava_spec_known_issues_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_known_issues_id_seq OWNED BY public.ava_spec_known_issues.id;


--
-- Name: ava_spec_performance_metrics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_performance_metrics (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    metric_name character varying(200) NOT NULL,
    metric_type character varying(100) NOT NULL,
    target_value numeric(15,4),
    target_unit character varying(50),
    current_value numeric(15,4),
    current_p50 numeric(15,4),
    current_p95 numeric(15,4),
    current_p99 numeric(15,4),
    meets_target boolean,
    historical_data jsonb DEFAULT '[]'::jsonb,
    measured_at timestamp with time zone DEFAULT now(),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_performance_metrics OWNER TO postgres;

--
-- Name: TABLE ava_spec_performance_metrics; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_performance_metrics IS 'Performance metrics and SLOs for each feature';


--
-- Name: ava_spec_performance_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_performance_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_performance_metrics_id_seq OWNER TO postgres;

--
-- Name: ava_spec_performance_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_performance_metrics_id_seq OWNED BY public.ava_spec_performance_metrics.id;


--
-- Name: ava_spec_source_files; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_source_files (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    file_path character varying(500) NOT NULL,
    file_type character varying(50) NOT NULL,
    is_primary boolean DEFAULT false,
    start_line integer,
    end_line integer,
    file_purpose character varying(500),
    key_exports text[],
    loc integer,
    complexity_score integer,
    last_modified timestamp with time zone,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_source_files OWNER TO postgres;

--
-- Name: TABLE ava_spec_source_files; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_source_files IS 'Source files associated with each feature specification';


--
-- Name: ava_spec_source_files_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_source_files_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_source_files_id_seq OWNER TO postgres;

--
-- Name: ava_spec_source_files_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_source_files_id_seq OWNED BY public.ava_spec_source_files.id;


--
-- Name: ava_spec_tags; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_tags (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    tag character varying(100) NOT NULL,
    tag_category character varying(100),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_tags OWNER TO postgres;

--
-- Name: TABLE ava_spec_tags; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_tags IS 'Tags for categorizing and searching features';


--
-- Name: ava_spec_tags_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_tags_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_tags_id_seq OWNER TO postgres;

--
-- Name: ava_spec_tags_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_tags_id_seq OWNED BY public.ava_spec_tags.id;


--
-- Name: ava_spec_version_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ava_spec_version_history (
    id integer NOT NULL,
    spec_id integer NOT NULL,
    version character varying(20) NOT NULL,
    previous_version character varying(20),
    change_type character varying(50) NOT NULL,
    change_summary text NOT NULL,
    change_details jsonb DEFAULT '{}'::jsonb,
    spec_snapshot jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.ava_spec_version_history OWNER TO postgres;

--
-- Name: TABLE ava_spec_version_history; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ava_spec_version_history IS 'Version history for tracking spec changes over time';


--
-- Name: ava_spec_version_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ava_spec_version_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_spec_version_history_id_seq OWNER TO postgres;

--
-- Name: ava_spec_version_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ava_spec_version_history_id_seq OWNED BY public.ava_spec_version_history.id;


--
-- Name: ava_user_goals; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.ava_user_goals (
    id integer NOT NULL,
    user_id character varying(100) DEFAULT 'default_user'::character varying NOT NULL,
    platform character varying(50) DEFAULT 'web'::character varying NOT NULL,
    goal_type character varying(50) NOT NULL,
    goal_name character varying(200) NOT NULL,
    target_value numeric(15,2) NOT NULL,
    target_unit character varying(50) NOT NULL,
    period_type character varying(20) NOT NULL,
    start_date date DEFAULT CURRENT_DATE NOT NULL,
    end_date date,
    current_value numeric(15,2) DEFAULT 0,
    progress_pct numeric(5,2) DEFAULT 0,
    last_updated_at timestamp with time zone,
    allowed_strategies text[],
    max_position_size numeric(10,2),
    max_total_exposure numeric(10,2),
    status character varying(20) DEFAULT 'active'::character varying,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_progress CHECK (((progress_pct >= (0)::numeric) AND (progress_pct <= (200)::numeric)))
);


ALTER TABLE public.ava_user_goals OWNER TO adam;

--
-- Name: TABLE ava_user_goals; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.ava_user_goals IS 'User income and return goals with progress tracking';


--
-- Name: COLUMN ava_user_goals.target_value; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_user_goals.target_value IS 'Target value (e.g., 2500 for $2,500/month income)';


--
-- Name: COLUMN ava_user_goals.progress_pct; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.ava_user_goals.progress_pct IS 'Progress percentage (can exceed 100% if goal is surpassed)';


--
-- Name: ava_user_goals_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.ava_user_goals_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ava_user_goals_id_seq OWNER TO adam;

--
-- Name: ava_user_goals_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.ava_user_goals_id_seq OWNED BY public.ava_user_goals.id;


--
-- Name: earnings_alerts; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.earnings_alerts (
    id integer NOT NULL,
    user_id uuid,
    symbol character varying(10) NOT NULL,
    alert_type character varying(30),
    days_before_earnings integer DEFAULT 1,
    min_surprise_percent numeric(8,2),
    min_iv_threshold numeric(6,4),
    is_active boolean DEFAULT true,
    notification_methods text[] DEFAULT ARRAY['email'::text],
    last_triggered_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT earnings_alerts_alert_type_check CHECK (((alert_type)::text = ANY ((ARRAY['upcoming_earnings'::character varying, 'earnings_beat'::character varying, 'earnings_miss'::character varying, 'high_iv_pre_earnings'::character varying, 'unusual_options_activity'::character varying])::text[])))
);


ALTER TABLE public.earnings_alerts OWNER TO adam;

--
-- Name: earnings_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.earnings_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_alerts_id_seq OWNER TO adam;

--
-- Name: earnings_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.earnings_alerts_id_seq OWNED BY public.earnings_alerts.id;


--
-- Name: earnings_events; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.earnings_events (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    earnings_date date NOT NULL,
    earnings_time character varying(10),
    fiscal_year integer,
    fiscal_quarter integer,
    eps_estimate numeric(10,4),
    revenue_estimate bigint,
    whisper_number numeric(10,4),
    eps_actual numeric(10,4),
    revenue_actual bigint,
    surprise_percent numeric(8,2),
    pre_earnings_price numeric(10,2),
    pre_earnings_iv numeric(6,4),
    pre_earnings_volume bigint,
    post_earnings_price numeric(10,2),
    post_earnings_iv numeric(6,4),
    post_earnings_volume bigint,
    price_move_percent numeric(8,2),
    volume_ratio numeric(8,2),
    options_volume integer,
    put_call_ratio numeric(6,4),
    unusual_options_activity boolean DEFAULT false,
    call_datetime timestamp with time zone,
    call_broadcast_url text,
    is_confirmed boolean DEFAULT false,
    has_occurred boolean DEFAULT false,
    data_source character varying(50) DEFAULT 'robinhood'::character varying,
    raw_data jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT earnings_events_earnings_time_check CHECK (((earnings_time)::text = ANY ((ARRAY['bmo'::character varying, 'amc'::character varying, 'unspecified'::character varying])::text[])))
);


ALTER TABLE public.earnings_events OWNER TO adam;

--
-- Name: earnings_calendar; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.earnings_calendar AS
 SELECT earnings_events.id,
    earnings_events.symbol,
    earnings_events.earnings_date AS report_date,
    earnings_events.earnings_time AS time_of_day,
    earnings_events.eps_estimate,
    earnings_events.revenue_estimate,
    earnings_events.is_confirmed,
    earnings_events.has_occurred
   FROM public.earnings_events;


ALTER TABLE public.earnings_calendar OWNER TO adam;

--
-- Name: earnings_events_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.earnings_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_events_id_seq OWNER TO adam;

--
-- Name: earnings_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.earnings_events_id_seq OWNED BY public.earnings_events.id;


--
-- Name: earnings_history; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.earnings_history (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    report_date date NOT NULL,
    fiscal_year integer,
    fiscal_quarter integer,
    earnings_time character varying(10),
    eps_actual numeric(10,4),
    eps_estimate numeric(10,4),
    eps_surprise numeric(10,4),
    eps_surprise_percent numeric(8,2),
    revenue_actual bigint,
    revenue_estimate bigint,
    revenue_surprise bigint,
    revenue_surprise_percent numeric(8,2),
    beat_miss character varying(10),
    call_datetime timestamp with time zone,
    call_broadcast_url text,
    data_source character varying(50) DEFAULT 'robinhood'::character varying,
    raw_data jsonb,
    synced_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT earnings_history_beat_miss_check CHECK (((beat_miss)::text = ANY ((ARRAY['beat'::character varying, 'miss'::character varying, 'meet'::character varying, 'unknown'::character varying])::text[]))),
    CONSTRAINT earnings_history_earnings_time_check CHECK (((earnings_time)::text = ANY ((ARRAY['bmo'::character varying, 'amc'::character varying, 'unspecified'::character varying])::text[])))
);


ALTER TABLE public.earnings_history OWNER TO adam;

--
-- Name: earnings_history_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.earnings_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_history_id_seq OWNER TO adam;

--
-- Name: earnings_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.earnings_history_id_seq OWNED BY public.earnings_history.id;


--
-- Name: earnings_sync_status; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.earnings_sync_status (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    last_sync_at timestamp with time zone,
    last_sync_status character varying(20),
    last_error_message text,
    historical_quarters_found integer DEFAULT 0,
    upcoming_events_found integer DEFAULT 0,
    total_syncs integer DEFAULT 0,
    failed_syncs integer DEFAULT 0,
    next_sync_at timestamp with time zone,
    sync_frequency_hours integer DEFAULT 24,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT earnings_sync_status_last_sync_status_check CHECK (((last_sync_status)::text = ANY ((ARRAY['success'::character varying, 'failed'::character varying, 'partial'::character varying, 'no_data'::character varying])::text[])))
);


ALTER TABLE public.earnings_sync_status OWNER TO adam;

--
-- Name: earnings_sync_status_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.earnings_sync_status_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.earnings_sync_status_id_seq OWNER TO adam;

--
-- Name: earnings_sync_status_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.earnings_sync_status_id_seq OWNED BY public.earnings_sync_status.id;


--
-- Name: etfs_universe; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.etfs_universe (
    id integer NOT NULL,
    symbol character varying(20) NOT NULL,
    fund_name character varying(255),
    exchange character varying(50),
    category character varying(100),
    fund_family character varying(150),
    current_price numeric(12,4),
    previous_close numeric(12,4),
    open_price numeric(12,4),
    day_high numeric(12,4),
    day_low numeric(12,4),
    week_52_high numeric(12,4),
    week_52_low numeric(12,4),
    nav_price numeric(12,4),
    volume bigint,
    avg_volume_10d bigint,
    avg_volume_3m bigint,
    total_assets bigint,
    expense_ratio numeric(12,6),
    yield_ttm numeric(12,6),
    ytd_return numeric(12,6),
    three_year_return numeric(12,6),
    five_year_return numeric(12,6),
    holdings_count integer,
    top_holding_symbol character varying(20),
    top_holding_weight numeric(12,6),
    beta numeric(12,6),
    sma_50 numeric(12,4),
    sma_200 numeric(12,4),
    rsi_14 numeric(12,4),
    dividend_yield numeric(12,6),
    dividend_rate numeric(12,4),
    ex_dividend_date date,
    has_options boolean DEFAULT false,
    implied_volatility numeric(12,6),
    data_source character varying(50) DEFAULT 'yfinance'::character varying,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_active boolean DEFAULT true
);


ALTER TABLE public.etfs_universe OWNER TO postgres;

--
-- Name: etfs_universe_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.etfs_universe_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.etfs_universe_id_seq OWNER TO postgres;

--
-- Name: etfs_universe_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.etfs_universe_id_seq OWNED BY public.etfs_universe.id;


--
-- Name: kalshi_markets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kalshi_markets (
    id integer NOT NULL,
    ticker character varying(100) NOT NULL,
    title text NOT NULL,
    subtitle text,
    market_type character varying(50) NOT NULL,
    series_ticker character varying(100),
    home_team character varying(100),
    away_team character varying(100),
    game_date timestamp with time zone,
    yes_price numeric(5,4),
    no_price numeric(5,4),
    volume numeric(15,2) DEFAULT 0,
    open_interest integer DEFAULT 0,
    status character varying(20) DEFAULT 'open'::character varying,
    close_time timestamp with time zone,
    expiration_time timestamp with time zone,
    result character varying(10),
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    synced_at timestamp with time zone DEFAULT now(),
    raw_data jsonb,
    CONSTRAINT chk_market_type CHECK (((market_type)::text = ANY ((ARRAY['nfl'::character varying, 'college'::character varying])::text[]))),
    CONSTRAINT chk_prices CHECK (((yes_price >= (0)::numeric) AND (yes_price <= (1)::numeric) AND (no_price >= (0)::numeric) AND (no_price <= (1)::numeric))),
    CONSTRAINT chk_result CHECK (((result)::text = ANY ((ARRAY['yes'::character varying, 'no'::character varying, NULL::character varying])::text[]))),
    CONSTRAINT chk_status CHECK (((status)::text = ANY ((ARRAY['open'::character varying, 'closed'::character varying, 'settled'::character varying])::text[])))
);


ALTER TABLE public.kalshi_markets OWNER TO postgres;

--
-- Name: TABLE kalshi_markets; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.kalshi_markets IS 'Stores all football market data from Kalshi prediction markets';


--
-- Name: COLUMN kalshi_markets.ticker; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_markets.ticker IS 'Unique Kalshi market ticker symbol';


--
-- Name: COLUMN kalshi_markets.market_type; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_markets.market_type IS 'Type of football: nfl or college';


--
-- Name: COLUMN kalshi_markets.yes_price; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_markets.yes_price IS 'Current market price for YES outcome (0-1)';


--
-- Name: COLUMN kalshi_markets.volume; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_markets.volume IS 'Total USD trading volume';


--
-- Name: COLUMN kalshi_markets.raw_data; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_markets.raw_data IS 'Complete API response for debugging';


--
-- Name: kalshi_markets_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kalshi_markets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kalshi_markets_id_seq OWNER TO postgres;

--
-- Name: kalshi_markets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kalshi_markets_id_seq OWNED BY public.kalshi_markets.id;


--
-- Name: kalshi_predictions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kalshi_predictions (
    id integer NOT NULL,
    market_id integer NOT NULL,
    ticker character varying(100) NOT NULL,
    predicted_outcome character varying(10),
    confidence_score numeric(5,2),
    edge_percentage numeric(5,2),
    overall_rank integer,
    type_rank integer,
    value_score numeric(5,2),
    liquidity_score numeric(5,2),
    timing_score numeric(5,2),
    matchup_score numeric(5,2),
    sentiment_score numeric(5,2),
    recommended_action character varying(20),
    recommended_stake_pct numeric(5,2),
    max_price numeric(5,4),
    reasoning text,
    key_factors jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_action CHECK (((recommended_action)::text = ANY ((ARRAY['strong_buy'::character varying, 'buy'::character varying, 'hold'::character varying, 'pass'::character varying])::text[]))),
    CONSTRAINT chk_confidence CHECK (((confidence_score >= (0)::numeric) AND (confidence_score <= (100)::numeric))),
    CONSTRAINT chk_predicted_outcome CHECK (((predicted_outcome)::text = ANY ((ARRAY['yes'::character varying, 'no'::character varying])::text[])))
);


ALTER TABLE public.kalshi_predictions OWNER TO postgres;

--
-- Name: TABLE kalshi_predictions; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.kalshi_predictions IS 'AI-generated predictions and rankings for Kalshi football markets';


--
-- Name: COLUMN kalshi_predictions.confidence_score; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_predictions.confidence_score IS 'AI confidence in prediction (0-100)';


--
-- Name: COLUMN kalshi_predictions.edge_percentage; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_predictions.edge_percentage IS 'Expected value edge over market price';


--
-- Name: COLUMN kalshi_predictions.recommended_stake_pct; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.kalshi_predictions.recommended_stake_pct IS 'Recommended bet size as % of bankroll';


--
-- Name: kalshi_predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kalshi_predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kalshi_predictions_id_seq OWNER TO postgres;

--
-- Name: kalshi_predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kalshi_predictions_id_seq OWNED BY public.kalshi_predictions.id;


--
-- Name: kalshi_price_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kalshi_price_history (
    id integer NOT NULL,
    market_id integer NOT NULL,
    ticker character varying(100) NOT NULL,
    yes_price numeric(5,4),
    no_price numeric(5,4),
    volume numeric(15,2),
    open_interest integer,
    snapshot_time timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_history_prices CHECK (((yes_price >= (0)::numeric) AND (yes_price <= (1)::numeric) AND (no_price >= (0)::numeric) AND (no_price <= (1)::numeric)))
);


ALTER TABLE public.kalshi_price_history OWNER TO postgres;

--
-- Name: TABLE kalshi_price_history; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.kalshi_price_history IS 'Historical price snapshots for charting and analysis';


--
-- Name: kalshi_price_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kalshi_price_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kalshi_price_history_id_seq OWNER TO postgres;

--
-- Name: kalshi_price_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kalshi_price_history_id_seq OWNED BY public.kalshi_price_history.id;


--
-- Name: kalshi_sync_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kalshi_sync_log (
    id integer NOT NULL,
    sync_type character varying(50) NOT NULL,
    market_type character varying(20),
    total_processed integer DEFAULT 0,
    successful integer DEFAULT 0,
    failed integer DEFAULT 0,
    duration_seconds integer,
    status character varying(20) DEFAULT 'running'::character varying,
    error_message text,
    started_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    CONSTRAINT chk_sync_status CHECK (((status)::text = ANY ((ARRAY['running'::character varying, 'completed'::character varying, 'error'::character varying])::text[]))),
    CONSTRAINT chk_sync_type CHECK (((sync_type)::text = ANY ((ARRAY['markets'::character varying, 'predictions'::character varying, 'prices'::character varying])::text[])))
);


ALTER TABLE public.kalshi_sync_log OWNER TO postgres;

--
-- Name: TABLE kalshi_sync_log; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.kalshi_sync_log IS 'Tracks Kalshi data sync operations';


--
-- Name: kalshi_sync_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kalshi_sync_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kalshi_sync_log_id_seq OWNER TO postgres;

--
-- Name: kalshi_sync_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kalshi_sync_log_id_seq OWNED BY public.kalshi_sync_log.id;


--
-- Name: live_odds_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_odds_snapshots (
    id integer NOT NULL,
    sport character varying(20) NOT NULL,
    game_id character varying(50) NOT NULL,
    snapshot_time timestamp with time zone DEFAULT now(),
    spread_home numeric(4,1),
    spread_odds_home integer,
    spread_odds_away integer,
    moneyline_home integer,
    moneyline_away integer,
    over_under numeric(5,1),
    over_odds integer,
    under_odds integer,
    home_score integer,
    away_score integer,
    period integer,
    time_remaining character varying(20),
    odds_source character varying(50)
);


ALTER TABLE public.live_odds_snapshots OWNER TO postgres;

--
-- Name: live_odds_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_odds_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.live_odds_snapshots_id_seq OWNER TO postgres;

--
-- Name: live_odds_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_odds_snapshots_id_seq OWNED BY public.live_odds_snapshots.id;


--
-- Name: live_prediction_snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_prediction_snapshots (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    sport character varying(20) NOT NULL,
    pregame_home_prob numeric(5,4),
    pregame_away_prob numeric(5,4),
    live_home_prob numeric(5,4),
    live_away_prob numeric(5,4),
    home_score integer,
    away_score integer,
    quarter_period integer,
    time_remaining character varying(10),
    possession character varying(10),
    momentum_score numeric(4,2),
    scoring_run character varying(20),
    snapshot_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.live_prediction_snapshots OWNER TO postgres;

--
-- Name: TABLE live_prediction_snapshots; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.live_prediction_snapshots IS 'Real-time prediction adjustments during live games';


--
-- Name: live_prediction_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_prediction_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.live_prediction_snapshots_id_seq OWNER TO postgres;

--
-- Name: live_prediction_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_prediction_snapshots_id_seq OWNED BY public.live_prediction_snapshots.id;


--
-- Name: model_performance; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.model_performance (
    id integer NOT NULL,
    sport character varying(20) NOT NULL,
    model_version character varying(20) NOT NULL,
    period_start date,
    period_end date,
    period_type character varying(20),
    total_predictions integer DEFAULT 0,
    correct_predictions integer DEFAULT 0,
    accuracy_rate numeric(5,4),
    brier_score numeric(6,5),
    log_loss numeric(8,5),
    theoretical_roi numeric(8,4),
    high_conf_roi numeric(8,4),
    high_conf_total integer DEFAULT 0,
    high_conf_correct integer DEFAULT 0,
    med_conf_total integer DEFAULT 0,
    med_conf_correct integer DEFAULT 0,
    low_conf_total integer DEFAULT 0,
    low_conf_correct integer DEFAULT 0,
    calculated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.model_performance OWNER TO postgres;

--
-- Name: TABLE model_performance; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.model_performance IS 'Aggregate performance metrics by time period and sport';


--
-- Name: model_performance_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.model_performance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.model_performance_id_seq OWNER TO postgres;

--
-- Name: model_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.model_performance_id_seq OWNED BY public.model_performance.id;


--
-- Name: nba_games; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nba_games (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    season character varying(10) NOT NULL,
    home_team character varying(100) NOT NULL,
    away_team character varying(100) NOT NULL,
    home_team_abbr character varying(10),
    away_team_abbr character varying(10),
    game_time timestamp with time zone NOT NULL,
    venue character varying(200),
    home_score integer DEFAULT 0,
    away_score integer DEFAULT 0,
    quarter integer DEFAULT 0,
    time_remaining character varying(20),
    possession character varying(10),
    game_status character varying(20) DEFAULT 'scheduled'::character varying,
    is_live boolean DEFAULT false,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    spread_home numeric(4,1),
    spread_odds_home integer,
    spread_odds_away integer,
    moneyline_home integer,
    moneyline_away integer,
    over_under numeric(5,1),
    over_odds integer,
    under_odds integer,
    opening_spread numeric(4,1),
    opening_total numeric(5,1),
    spread_movement numeric(4,1),
    total_movement numeric(4,1),
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    last_synced timestamp with time zone DEFAULT now(),
    raw_game_data jsonb,
    CONSTRAINT chk_nba_game_status CHECK (((game_status)::text = ANY ((ARRAY['scheduled'::character varying, 'live'::character varying, 'halftime'::character varying, 'final'::character varying, 'postponed'::character varying, 'cancelled'::character varying, 'delayed'::character varying])::text[])))
);


ALTER TABLE public.nba_games OWNER TO postgres;

--
-- Name: nba_games_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nba_games_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nba_games_id_seq OWNER TO postgres;

--
-- Name: nba_games_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nba_games_id_seq OWNED BY public.nba_games.id;


--
-- Name: ncaa_basketball_games; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ncaa_basketball_games (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    season character varying(10) NOT NULL,
    home_team character varying(100) NOT NULL,
    away_team character varying(100) NOT NULL,
    home_team_abbr character varying(20),
    away_team_abbr character varying(20),
    home_rank integer,
    away_rank integer,
    conference character varying(50),
    game_time timestamp with time zone NOT NULL,
    venue character varying(200),
    home_score integer DEFAULT 0,
    away_score integer DEFAULT 0,
    half integer DEFAULT 0,
    time_remaining character varying(20),
    possession character varying(20),
    game_status character varying(20) DEFAULT 'scheduled'::character varying,
    is_live boolean DEFAULT false,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    spread_home numeric(4,1),
    spread_odds_home integer,
    spread_odds_away integer,
    moneyline_home integer,
    moneyline_away integer,
    over_under numeric(5,1),
    over_odds integer,
    under_odds integer,
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    last_synced timestamp with time zone DEFAULT now(),
    raw_game_data jsonb,
    CONSTRAINT chk_ncaab_game_status CHECK (((game_status)::text = ANY ((ARRAY['scheduled'::character varying, 'live'::character varying, 'halftime'::character varying, 'final'::character varying, 'postponed'::character varying, 'cancelled'::character varying, 'delayed'::character varying])::text[])))
);


ALTER TABLE public.ncaa_basketball_games OWNER TO postgres;

--
-- Name: ncaa_basketball_games_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ncaa_basketball_games_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ncaa_basketball_games_id_seq OWNER TO postgres;

--
-- Name: ncaa_basketball_games_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ncaa_basketball_games_id_seq OWNED BY public.ncaa_basketball_games.id;


--
-- Name: ncaa_football_games; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ncaa_football_games (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    home_team character varying(100) NOT NULL,
    away_team character varying(100) NOT NULL,
    home_team_abbr character varying(20),
    away_team_abbr character varying(20),
    home_rank integer,
    away_rank integer,
    conference character varying(50),
    game_time timestamp with time zone NOT NULL,
    venue character varying(200),
    is_outdoor boolean DEFAULT true,
    home_score integer DEFAULT 0,
    away_score integer DEFAULT 0,
    quarter integer DEFAULT 0,
    time_remaining character varying(20),
    possession character varying(20),
    game_status character varying(20) DEFAULT 'scheduled'::character varying,
    is_live boolean DEFAULT false,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    spread_home numeric(4,1),
    spread_odds_home integer,
    spread_odds_away integer,
    moneyline_home integer,
    moneyline_away integer,
    over_under numeric(5,1),
    over_odds integer,
    under_odds integer,
    opening_spread numeric(4,1),
    opening_total numeric(5,1),
    temperature integer,
    weather_condition character varying(100),
    wind_speed integer,
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    last_synced timestamp with time zone DEFAULT now(),
    raw_game_data jsonb,
    CONSTRAINT chk_ncaaf_game_status CHECK (((game_status)::text = ANY ((ARRAY['scheduled'::character varying, 'live'::character varying, 'halftime'::character varying, 'final'::character varying, 'postponed'::character varying, 'cancelled'::character varying, 'delayed'::character varying])::text[])))
);


ALTER TABLE public.ncaa_football_games OWNER TO postgres;

--
-- Name: ncaa_football_games_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ncaa_football_games_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ncaa_football_games_id_seq OWNER TO postgres;

--
-- Name: ncaa_football_games_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ncaa_football_games_id_seq OWNED BY public.ncaa_football_games.id;


--
-- Name: nfl_alert_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_alert_history (
    id integer NOT NULL,
    trigger_id integer,
    alert_type character varying(50) NOT NULL,
    subject character varying(500) NOT NULL,
    message text NOT NULL,
    game_id integer,
    play_id integer,
    kalshi_market_id integer,
    notification_channel character varying(50) NOT NULL,
    sent_at timestamp with time zone DEFAULT now(),
    delivery_status character varying(20) DEFAULT 'pending'::character varying,
    telegram_message_id character varying(100),
    error_message text,
    alert_data jsonb,
    CONSTRAINT chk_delivery_status CHECK (((delivery_status)::text = ANY ((ARRAY['pending'::character varying, 'sent'::character varying, 'failed'::character varying, 'queued'::character varying])::text[])))
);


ALTER TABLE public.nfl_alert_history OWNER TO postgres;

--
-- Name: TABLE nfl_alert_history; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_alert_history IS 'Historical record of all alerts sent';


--
-- Name: nfl_alert_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_alert_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_alert_history_id_seq OWNER TO postgres;

--
-- Name: nfl_alert_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_alert_history_id_seq OWNED BY public.nfl_alert_history.id;


--
-- Name: nfl_alert_triggers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_alert_triggers (
    id integer NOT NULL,
    alert_name character varying(200) NOT NULL,
    alert_type character varying(50) NOT NULL,
    is_active boolean DEFAULT true,
    trigger_conditions jsonb NOT NULL,
    teams_filter character varying(100)[],
    players_filter character varying(200)[],
    notification_channels character varying(50)[] DEFAULT ARRAY['telegram'::text],
    notification_priority character varying(20) DEFAULT 'medium'::character varying,
    cooldown_minutes integer DEFAULT 5,
    max_alerts_per_day integer DEFAULT 50,
    created_by character varying(100),
    created_at timestamp with time zone DEFAULT now(),
    last_triggered timestamp with time zone,
    trigger_count integer DEFAULT 0,
    CONSTRAINT chk_notification_priority CHECK (((notification_priority)::text = ANY ((ARRAY['low'::character varying, 'medium'::character varying, 'high'::character varying, 'urgent'::character varying])::text[])))
);


ALTER TABLE public.nfl_alert_triggers OWNER TO postgres;

--
-- Name: TABLE nfl_alert_triggers; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_alert_triggers IS 'User-defined alert conditions and preferences';


--
-- Name: nfl_alert_triggers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_alert_triggers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_alert_triggers_id_seq OWNER TO postgres;

--
-- Name: nfl_alert_triggers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_alert_triggers_id_seq OWNED BY public.nfl_alert_triggers.id;


--
-- Name: nfl_data_sync_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_data_sync_log (
    id integer NOT NULL,
    sync_type character varying(50) NOT NULL,
    sync_scope character varying(100),
    records_fetched integer DEFAULT 0,
    records_inserted integer DEFAULT 0,
    records_updated integer DEFAULT 0,
    records_failed integer DEFAULT 0,
    duration_ms integer,
    api_calls_made integer DEFAULT 0,
    api_errors integer DEFAULT 0,
    rate_limit_hit boolean DEFAULT false,
    sync_status character varying(20) DEFAULT 'running'::character varying,
    error_message text,
    started_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    sync_metadata jsonb,
    CONSTRAINT chk_sync_status CHECK (((sync_status)::text = ANY ((ARRAY['running'::character varying, 'completed'::character varying, 'failed'::character varying, 'partial'::character varying])::text[])))
);


ALTER TABLE public.nfl_data_sync_log OWNER TO postgres;

--
-- Name: TABLE nfl_data_sync_log; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_data_sync_log IS 'Performance tracking for all data sync operations';


--
-- Name: nfl_data_sync_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_data_sync_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_data_sync_log_id_seq OWNER TO postgres;

--
-- Name: nfl_data_sync_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_data_sync_log_id_seq OWNED BY public.nfl_data_sync_log.id;


--
-- Name: nfl_games; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_games (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    home_team character varying(50) NOT NULL,
    away_team character varying(50) NOT NULL,
    home_team_abbr character varying(10),
    away_team_abbr character varying(10),
    game_time timestamp with time zone NOT NULL,
    venue character varying(200),
    is_outdoor boolean DEFAULT true,
    home_score integer DEFAULT 0,
    away_score integer DEFAULT 0,
    quarter integer DEFAULT 0,
    time_remaining character varying(20),
    possession character varying(10),
    game_status character varying(20) DEFAULT 'scheduled'::character varying,
    is_live boolean DEFAULT false,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    spread_home numeric(4,1),
    spread_odds_home integer,
    spread_odds_away integer,
    moneyline_home integer,
    moneyline_away integer,
    over_under numeric(4,1),
    over_odds integer,
    under_odds integer,
    temperature integer,
    weather_condition character varying(100),
    wind_speed integer,
    precipitation_chance integer,
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    last_synced timestamp with time zone DEFAULT now(),
    raw_game_data jsonb,
    raw_weather_data jsonb,
    CONSTRAINT chk_game_status CHECK (((game_status)::text = ANY ((ARRAY['scheduled'::character varying, 'live'::character varying, 'halftime'::character varying, 'final'::character varying, 'postponed'::character varying, 'cancelled'::character varying])::text[]))),
    CONSTRAINT chk_quarter CHECK (((quarter >= 0) AND (quarter <= 5)))
);


ALTER TABLE public.nfl_games OWNER TO postgres;

--
-- Name: TABLE nfl_games; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_games IS 'NFL game schedules and live scores';


--
-- Name: COLUMN nfl_games.game_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.nfl_games.game_id IS 'Unique identifier from external API';


--
-- Name: COLUMN nfl_games.is_live; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.nfl_games.is_live IS 'Quick filter for active games';


--
-- Name: nfl_games_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_games_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_games_id_seq OWNER TO postgres;

--
-- Name: nfl_games_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_games_id_seq OWNED BY public.nfl_games.id;


--
-- Name: nfl_injuries; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_injuries (
    id integer NOT NULL,
    player_name character varying(200) NOT NULL,
    player_id character varying(100),
    team character varying(50) NOT NULL,
    "position" character varying(20),
    injury_type character varying(200),
    injury_status character varying(50) NOT NULL,
    description text,
    game_id integer,
    week integer,
    reported_at timestamp with time zone NOT NULL,
    resolved_at timestamp with time zone,
    is_active boolean DEFAULT true,
    source character varying(100),
    created_at timestamp with time zone DEFAULT now(),
    last_updated timestamp with time zone DEFAULT now(),
    raw_injury_data jsonb,
    CONSTRAINT chk_injury_status CHECK (((injury_status)::text = ANY ((ARRAY['Out'::character varying, 'Questionable'::character varying, 'Doubtful'::character varying, 'Probable'::character varying, 'IR'::character varying, 'PUP'::character varying, 'Cleared'::character varying])::text[])))
);


ALTER TABLE public.nfl_injuries OWNER TO postgres;

--
-- Name: TABLE nfl_injuries; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_injuries IS 'Player injury tracking and status updates';


--
-- Name: nfl_injuries_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_injuries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_injuries_id_seq OWNER TO postgres;

--
-- Name: nfl_injuries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_injuries_id_seq OWNED BY public.nfl_injuries.id;


--
-- Name: nfl_kalshi_correlations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_kalshi_correlations (
    id integer NOT NULL,
    game_id integer NOT NULL,
    play_id integer,
    event_type character varying(50) NOT NULL,
    event_timestamp timestamp with time zone NOT NULL,
    kalshi_market_id integer NOT NULL,
    market_ticker character varying(100) NOT NULL,
    price_before numeric(5,4),
    price_after numeric(5,4),
    price_change_pct numeric(6,2),
    volume_before numeric(15,2),
    volume_after numeric(15,2),
    volume_spike_pct numeric(6,2),
    correlation_strength numeric(4,3),
    impact_level character varying(20),
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT chk_impact_level CHECK (((impact_level)::text = ANY ((ARRAY['low'::character varying, 'medium'::character varying, 'high'::character varying, 'extreme'::character varying])::text[])))
);


ALTER TABLE public.nfl_kalshi_correlations OWNER TO postgres;

--
-- Name: TABLE nfl_kalshi_correlations; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_kalshi_correlations IS 'Correlation between NFL events and Kalshi market movements';


--
-- Name: nfl_kalshi_correlations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_kalshi_correlations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_kalshi_correlations_id_seq OWNER TO postgres;

--
-- Name: nfl_kalshi_correlations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_kalshi_correlations_id_seq OWNED BY public.nfl_kalshi_correlations.id;


--
-- Name: nfl_player_stats; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_player_stats (
    id integer NOT NULL,
    game_id integer NOT NULL,
    player_name character varying(200) NOT NULL,
    player_id character varying(100),
    team character varying(50) NOT NULL,
    "position" character varying(20) NOT NULL,
    passing_attempts integer DEFAULT 0,
    passing_completions integer DEFAULT 0,
    passing_yards integer DEFAULT 0,
    passing_touchdowns integer DEFAULT 0,
    passing_interceptions integer DEFAULT 0,
    rushing_attempts integer DEFAULT 0,
    rushing_yards integer DEFAULT 0,
    rushing_touchdowns integer DEFAULT 0,
    receptions integer DEFAULT 0,
    receiving_yards integer DEFAULT 0,
    receiving_touchdowns integer DEFAULT 0,
    targets integer DEFAULT 0,
    tackles integer DEFAULT 0,
    sacks numeric(3,1) DEFAULT 0,
    interceptions integer DEFAULT 0,
    forced_fumbles integer DEFAULT 0,
    field_goals_made integer DEFAULT 0,
    field_goals_attempted integer DEFAULT 0,
    extra_points_made integer DEFAULT 0,
    extra_points_attempted integer DEFAULT 0,
    last_updated timestamp with time zone DEFAULT now(),
    raw_stats_data jsonb
);


ALTER TABLE public.nfl_player_stats OWNER TO postgres;

--
-- Name: TABLE nfl_player_stats; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_player_stats IS 'Real-time player statistics for each game';


--
-- Name: nfl_player_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_player_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_player_stats_id_seq OWNER TO postgres;

--
-- Name: nfl_player_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_player_stats_id_seq OWNED BY public.nfl_player_stats.id;


--
-- Name: nfl_plays; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_plays (
    id integer NOT NULL,
    game_id integer NOT NULL,
    play_id character varying(100) NOT NULL,
    sequence_number integer NOT NULL,
    quarter integer NOT NULL,
    time_remaining character varying(20),
    play_type character varying(50),
    description text,
    down integer,
    yards_to_go integer,
    yard_line integer,
    yards_gained integer,
    is_scoring_play boolean DEFAULT false,
    is_turnover boolean DEFAULT false,
    is_penalty boolean DEFAULT false,
    offense_team character varying(10),
    defense_team character varying(10),
    player_name character varying(200),
    player_position character varying(10),
    points_home integer DEFAULT 0,
    points_away integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT now(),
    raw_play_data jsonb,
    CONSTRAINT chk_down CHECK ((((down >= 1) AND (down <= 4)) OR (down IS NULL)))
);


ALTER TABLE public.nfl_plays OWNER TO postgres;

--
-- Name: TABLE nfl_plays; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_plays IS 'Play-by-play data for NFL games';


--
-- Name: nfl_plays_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_plays_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_plays_id_seq OWNER TO postgres;

--
-- Name: nfl_plays_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_plays_id_seq OWNED BY public.nfl_plays.id;


--
-- Name: nfl_social_sentiment; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nfl_social_sentiment (
    id integer NOT NULL,
    game_id integer,
    entity_type character varying(20) NOT NULL,
    entity_id character varying(200) NOT NULL,
    sentiment_score numeric(4,3),
    positive_count integer DEFAULT 0,
    negative_count integer DEFAULT 0,
    neutral_count integer DEFAULT 0,
    total_mentions integer DEFAULT 0,
    twitter_mentions integer DEFAULT 0,
    reddit_mentions integer DEFAULT 0,
    window_start timestamp with time zone NOT NULL,
    window_end timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    raw_sentiment_data jsonb,
    CONSTRAINT chk_entity_type CHECK (((entity_type)::text = ANY ((ARRAY['game'::character varying, 'team'::character varying, 'player'::character varying])::text[]))),
    CONSTRAINT chk_sentiment_score CHECK (((sentiment_score >= '-1.0'::numeric) AND (sentiment_score <= 1.0)))
);


ALTER TABLE public.nfl_social_sentiment OWNER TO postgres;

--
-- Name: TABLE nfl_social_sentiment; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.nfl_social_sentiment IS 'Social media sentiment analysis for games and players';


--
-- Name: nfl_social_sentiment_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.nfl_social_sentiment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.nfl_social_sentiment_id_seq OWNER TO postgres;

--
-- Name: nfl_social_sentiment_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.nfl_social_sentiment_id_seq OWNED BY public.nfl_social_sentiment.id;


--
-- Name: odds_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.odds_history (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    sport character varying(20) NOT NULL,
    source character varying(30) NOT NULL,
    home_odds integer,
    away_odds integer,
    spread numeric(4,1),
    spread_home_odds integer DEFAULT '-110'::integer,
    spread_away_odds integer DEFAULT '-110'::integer,
    total numeric(5,1),
    over_odds integer DEFAULT '-110'::integer,
    under_odds integer DEFAULT '-110'::integer,
    home_implied_prob numeric(5,4),
    away_implied_prob numeric(5,4),
    recorded_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.odds_history OWNER TO postgres;

--
-- Name: TABLE odds_history; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.odds_history IS 'Historical odds data for movement charts and trend analysis';


--
-- Name: odds_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.odds_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.odds_history_id_seq OWNER TO postgres;

--
-- Name: odds_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.odds_history_id_seq OWNED BY public.odds_history.id;


--
-- Name: portfolio_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.portfolio_history (
    id integer NOT NULL,
    date date NOT NULL,
    portfolio_value numeric(15,2) NOT NULL,
    day_change numeric(15,2) DEFAULT 0,
    day_change_pct numeric(8,4) DEFAULT 0,
    buying_power numeric(15,2) DEFAULT 0,
    stocks_value numeric(15,2) DEFAULT 0,
    options_value numeric(15,2) DEFAULT 0,
    cash_value numeric(15,2) DEFAULT 0,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.portfolio_history OWNER TO postgres;

--
-- Name: portfolio_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.portfolio_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.portfolio_history_id_seq OWNER TO postgres;

--
-- Name: portfolio_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.portfolio_history_id_seq OWNED BY public.portfolio_history.id;


--
-- Name: prediction_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prediction_results (
    id integer NOT NULL,
    prediction_id character varying(100),
    game_id character varying(50) NOT NULL,
    sport character varying(20) NOT NULL,
    predicted_winner character varying(100),
    predicted_probability numeric(5,4),
    predicted_spread numeric(4,1),
    predicted_total numeric(5,1),
    actual_winner character varying(100),
    actual_home_score integer,
    actual_away_score integer,
    was_correct boolean,
    prediction_timestamp timestamp without time zone DEFAULT now(),
    game_completed_at timestamp without time zone,
    model_version character varying(20) DEFAULT 'v1.0'::character varying,
    confidence_tier character varying(20),
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.prediction_results OWNER TO postgres;

--
-- Name: TABLE prediction_results; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.prediction_results IS 'Stores all AI predictions and their outcomes for accuracy tracking';


--
-- Name: prediction_results_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.prediction_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.prediction_results_id_seq OWNER TO postgres;

--
-- Name: prediction_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.prediction_results_id_seq OWNED BY public.prediction_results.id;


--
-- Name: premium_opportunities; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.premium_opportunities (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    company_name character varying(255),
    option_type character varying(4) NOT NULL,
    strike numeric(10,2) NOT NULL,
    expiration date NOT NULL,
    dte integer NOT NULL,
    stock_price numeric(10,2),
    bid numeric(10,4),
    ask numeric(10,4),
    mid numeric(10,4),
    premium numeric(10,4),
    premium_pct numeric(6,3),
    annualized_return numeric(8,3),
    monthly_return numeric(8,3),
    delta numeric(8,5),
    gamma numeric(8,5),
    theta numeric(10,5),
    vega numeric(10,5),
    rho numeric(10,5),
    implied_volatility numeric(8,4),
    volume integer,
    open_interest integer,
    break_even numeric(10,2),
    max_profit numeric(10,2),
    max_loss numeric(10,2),
    pop numeric(6,3),
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    scan_id character varying(50)
);


ALTER TABLE public.premium_opportunities OWNER TO postgres;

--
-- Name: TABLE premium_opportunities; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.premium_opportunities IS 'Stores latest premium opportunities per option contract, updated periodically';


--
-- Name: premium_opportunities_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.premium_opportunities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.premium_opportunities_id_seq OWNER TO postgres;

--
-- Name: premium_opportunities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.premium_opportunities_id_seq OWNED BY public.premium_opportunities.id;


--
-- Name: premium_scan_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.premium_scan_history (
    id integer NOT NULL,
    scan_id character varying(50) NOT NULL,
    symbols text[],
    symbol_count integer,
    dte integer,
    max_price numeric(10,2),
    min_premium_pct numeric(5,2),
    result_count integer,
    results jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.premium_scan_history OWNER TO postgres;

--
-- Name: TABLE premium_scan_history; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.premium_scan_history IS 'Stores history of premium scans with full results';


--
-- Name: premium_scan_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.premium_scan_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.premium_scan_history_id_seq OWNER TO postgres;

--
-- Name: premium_scan_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.premium_scan_history_id_seq OWNED BY public.premium_scan_history.id;


--
-- Name: scanner_results; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.scanner_results (
    id integer NOT NULL,
    symbol character varying(20) NOT NULL,
    strike numeric(10,2),
    expiration_date date,
    premium numeric(10,2),
    annual_return numeric(10,2),
    delta numeric(6,4),
    scan_type character varying(20) DEFAULT 'CSP'::character varying,
    scanned_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.scanner_results OWNER TO adam;

--
-- Name: TABLE scanner_results; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.scanner_results IS 'Results from premium scanner for CSP/CC opportunities';


--
-- Name: scanner_results_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.scanner_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.scanner_results_id_seq OWNER TO adam;

--
-- Name: scanner_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.scanner_results_id_seq OWNED BY public.scanner_results.id;


--
-- Name: scanner_watchlists; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scanner_watchlists (
    id integer NOT NULL,
    watchlist_id character varying(100) NOT NULL,
    source character varying(50) NOT NULL,
    name character varying(255) NOT NULL,
    symbols text[] NOT NULL,
    symbol_count integer GENERATED ALWAYS AS (array_length(symbols, 1)) STORED,
    category character varying(100),
    sort_order integer DEFAULT 1000,
    is_active boolean DEFAULT true,
    last_synced timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.scanner_watchlists OWNER TO postgres;

--
-- Name: TABLE scanner_watchlists; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.scanner_watchlists IS 'Cached watchlists for Premium Scanner - synced periodically for fast API responses';


--
-- Name: COLUMN scanner_watchlists.watchlist_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.scanner_watchlists.watchlist_id IS 'Unique identifier: source_name format (e.g., predefined_popular, tv_123456)';


--
-- Name: COLUMN scanner_watchlists.source; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.scanner_watchlists.source IS 'Data source: predefined, database, tradingview, robinhood';


--
-- Name: COLUMN scanner_watchlists.symbols; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.scanner_watchlists.symbols IS 'Array of stock symbols in this watchlist';


--
-- Name: COLUMN scanner_watchlists.sort_order; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.scanner_watchlists.sort_order IS 'Lower numbers appear first in dropdown';


--
-- Name: scanner_watchlists_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.scanner_watchlists_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.scanner_watchlists_id_seq OWNER TO postgres;

--
-- Name: scanner_watchlists_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.scanner_watchlists_id_seq OWNED BY public.scanner_watchlists.id;


--
-- Name: scanner_watchlists_sync_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scanner_watchlists_sync_log (
    id integer NOT NULL,
    sync_type character varying(50) NOT NULL,
    source character varying(50),
    watchlists_synced integer DEFAULT 0,
    total_symbols integer DEFAULT 0,
    duration_seconds numeric(10,2),
    status character varying(20) DEFAULT 'success'::character varying,
    error_message text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.scanner_watchlists_sync_log OWNER TO postgres;

--
-- Name: scanner_watchlists_sync_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.scanner_watchlists_sync_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.scanner_watchlists_sync_log_id_seq OWNER TO postgres;

--
-- Name: scanner_watchlists_sync_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.scanner_watchlists_sync_log_id_seq OWNED BY public.scanner_watchlists_sync_log.id;


--
-- Name: stock_ai_scores; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stock_ai_scores (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    company_name character varying(255),
    sector character varying(100),
    ai_score numeric(5,2) NOT NULL,
    recommendation character varying(20) NOT NULL,
    confidence numeric(4,3) NOT NULL,
    current_price numeric(12,4) NOT NULL,
    daily_change_pct numeric(6,3),
    trend character varying(20),
    trend_strength numeric(4,3),
    rsi_14 numeric(5,2),
    iv_estimate numeric(6,2),
    vol_regime character varying(20),
    predicted_change_1d numeric(6,3),
    predicted_change_5d numeric(6,3),
    support_price numeric(12,4),
    resistance_price numeric(12,4),
    score_components jsonb DEFAULT '{}'::jsonb,
    scored_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.stock_ai_scores OWNER TO postgres;

--
-- Name: stock_ai_scores_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.stock_ai_scores_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.stock_ai_scores_id_seq OWNER TO postgres;

--
-- Name: stock_ai_scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.stock_ai_scores_id_seq OWNED BY public.stock_ai_scores.id;


--
-- Name: stocks_universe; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stocks_universe (
    id integer NOT NULL,
    symbol character varying(20) NOT NULL,
    company_name character varying(255),
    exchange character varying(50),
    sector character varying(100),
    industry character varying(150),
    current_price numeric(12,4),
    previous_close numeric(12,4),
    open_price numeric(12,4),
    day_high numeric(12,4),
    day_low numeric(12,4),
    week_52_high numeric(12,4),
    week_52_low numeric(12,4),
    volume bigint,
    avg_volume_10d bigint,
    avg_volume_3m bigint,
    market_cap bigint,
    shares_outstanding bigint,
    float_shares bigint,
    pe_ratio numeric(12,4),
    forward_pe numeric(12,4),
    peg_ratio numeric(12,4),
    price_to_book numeric(12,4),
    price_to_sales numeric(12,4),
    enterprise_value bigint,
    ev_to_ebitda numeric(12,4),
    ev_to_revenue numeric(12,4),
    profit_margin numeric(12,6),
    operating_margin numeric(12,6),
    gross_margin numeric(12,6),
    roe numeric(12,6),
    roa numeric(12,6),
    revenue_growth numeric(12,6),
    earnings_growth numeric(12,6),
    quarterly_revenue_growth numeric(12,6),
    quarterly_earnings_growth numeric(12,6),
    dividend_yield numeric(12,6),
    dividend_rate numeric(12,4),
    payout_ratio numeric(12,6),
    ex_dividend_date date,
    total_cash bigint,
    total_debt bigint,
    debt_to_equity numeric(12,4),
    current_ratio numeric(12,4),
    quick_ratio numeric(12,4),
    free_cash_flow bigint,
    beta numeric(12,6),
    sma_50 numeric(12,4),
    sma_200 numeric(12,4),
    rsi_14 numeric(12,4),
    target_high_price numeric(12,4),
    target_low_price numeric(12,4),
    target_mean_price numeric(12,4),
    recommendation_key character varying(50),
    number_of_analysts integer,
    has_options boolean DEFAULT false,
    implied_volatility numeric(12,6),
    data_source character varying(50) DEFAULT 'yfinance'::character varying,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_active boolean DEFAULT true
);


ALTER TABLE public.stocks_universe OWNER TO postgres;

--
-- Name: stocks_universe_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.stocks_universe_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.stocks_universe_id_seq OWNER TO postgres;

--
-- Name: stocks_universe_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.stocks_universe_id_seq OWNED BY public.stocks_universe.id;


--
-- Name: trade_history; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.trade_history (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    strategy_type character varying(20) DEFAULT 'cash_secured_put'::character varying,
    open_date timestamp with time zone DEFAULT now() NOT NULL,
    strike_price numeric(10,2) NOT NULL,
    expiration_date date NOT NULL,
    premium_collected numeric(10,2) NOT NULL,
    contracts integer DEFAULT 1,
    dte_at_open integer,
    close_date timestamp with time zone,
    close_price numeric(10,2),
    close_reason character varying(20),
    days_held integer,
    profit_loss numeric(10,2),
    profit_loss_percent numeric(10,4),
    annualized_return numeric(10,4),
    status character varying(20) DEFAULT 'open'::character varying,
    notes text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.trade_history OWNER TO adam;

--
-- Name: TABLE trade_history; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.trade_history IS 'Tracks all option trades (open and closed positions)';


--
-- Name: COLUMN trade_history.strategy_type; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.trade_history.strategy_type IS 'Type of strategy: cash_secured_put, covered_call, etc';


--
-- Name: COLUMN trade_history.close_reason; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.trade_history.close_reason IS 'Why position was closed: early_close, expiration, assignment';


--
-- Name: COLUMN trade_history.profit_loss; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.trade_history.profit_loss IS 'Net profit/loss in dollars';


--
-- Name: COLUMN trade_history.annualized_return; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.trade_history.annualized_return IS 'Annualized return percentage';


--
-- Name: trade_history_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.trade_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trade_history_id_seq OWNER TO adam;

--
-- Name: trade_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.trade_history_id_seq OWNED BY public.trade_history.id;


--
-- Name: trade_journal; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.trade_journal (
    id integer NOT NULL,
    symbol character varying(20) NOT NULL,
    strategy character varying(50),
    asset_type character varying(20) DEFAULT 'option'::character varying,
    status character varying(20) DEFAULT 'open'::character varying,
    entry_price numeric(12,4),
    quantity integer DEFAULT 1,
    realized_pnl numeric(12,2),
    opened_at timestamp with time zone DEFAULT now(),
    closed_at timestamp with time zone,
    notes text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.trade_journal OWNER TO adam;

--
-- Name: TABLE trade_journal; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.trade_journal IS 'Comprehensive trade journal for P&L tracking';


--
-- Name: trade_journal_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.trade_journal_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trade_journal_id_seq OWNER TO adam;

--
-- Name: trade_journal_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.trade_journal_id_seq OWNED BY public.trade_journal.id;


--
-- Name: tv_symbols_api; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tv_symbols_api (
    id integer NOT NULL,
    watchlist_id character varying(50),
    symbol character varying(50) NOT NULL,
    exchange character varying(50),
    full_symbol character varying(100),
    added_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.tv_symbols_api OWNER TO postgres;

--
-- Name: tv_symbols_api_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.tv_symbols_api_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.tv_symbols_api_id_seq OWNER TO postgres;

--
-- Name: tv_symbols_api_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.tv_symbols_api_id_seq OWNED BY public.tv_symbols_api.id;


--
-- Name: tv_watchlists_api; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tv_watchlists_api (
    id integer NOT NULL,
    watchlist_id character varying(50) NOT NULL,
    name character varying(255) NOT NULL,
    color character varying(50),
    symbols text[],
    symbol_count integer DEFAULT 0,
    last_synced timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.tv_watchlists_api OWNER TO postgres;

--
-- Name: tv_watchlists_api_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.tv_watchlists_api_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.tv_watchlists_api_id_seq OWNER TO postgres;

--
-- Name: tv_watchlists_api_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.tv_watchlists_api_id_seq OWNED BY public.tv_watchlists_api.id;


--
-- Name: user_bets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_bets (
    id integer NOT NULL,
    user_id character varying(50),
    prediction_id character varying(100),
    game_id character varying(50) NOT NULL,
    sport character varying(20) NOT NULL,
    bet_type character varying(30),
    bet_side character varying(50),
    bet_amount numeric(12,2),
    odds integer,
    potential_payout numeric(12,2),
    actual_payout numeric(12,2),
    is_winner boolean,
    ai_confidence numeric(5,4),
    ai_recommended boolean DEFAULT false,
    placed_at timestamp without time zone DEFAULT now(),
    settled_at timestamp without time zone
);


ALTER TABLE public.user_bets OWNER TO postgres;

--
-- Name: user_bets_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_bets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_bets_id_seq OWNER TO postgres;

--
-- Name: user_bets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_bets_id_seq OWNED BY public.user_bets.id;


--
-- Name: user_betting_profile; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_betting_profile (
    user_id character varying(50) NOT NULL,
    preferred_sports text[] DEFAULT ARRAY['NFL'::text, 'NBA'::text],
    preferred_bet_types text[] DEFAULT ARRAY['moneyline'::text, 'spread'::text],
    risk_tolerance character varying(20) DEFAULT 'medium'::character varying,
    bankroll numeric(12,2) DEFAULT 1000.00,
    kelly_fraction numeric(3,2) DEFAULT 0.25,
    max_bet_size numeric(12,2) DEFAULT 100.00,
    total_bets integer DEFAULT 0,
    total_wins integer DEFAULT 0,
    total_profit numeric(12,2) DEFAULT 0.00,
    notifications_enabled boolean DEFAULT true,
    confidence_threshold numeric(3,2) DEFAULT 0.60,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.user_betting_profile OWNER TO postgres;

--
-- Name: v_automation_status; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_automation_status AS
 SELECT a.id,
    a.name,
    a.display_name,
    a.automation_type,
    a.celery_task_name,
    a.schedule_type,
    a.schedule_config,
    a.schedule_display,
    a.queue,
    a.category,
    a.description,
    a.is_enabled,
    a.enabled_updated_at,
    a.enabled_updated_by,
    a.timeout_seconds,
    a.max_retries,
    a.created_at,
    a.updated_at,
    latest.id AS last_execution_id,
    latest.status AS last_run_status,
    latest.started_at AS last_run_at,
    latest.completed_at AS last_completed_at,
    latest.duration_seconds AS last_duration_seconds,
    latest.error_message AS last_error,
    latest.records_processed AS last_records_processed,
    NULL::timestamp with time zone AS next_run_at,
    stats.total_runs_24h,
    stats.successful_runs_24h,
    stats.failed_runs_24h,
        CASE
            WHEN (stats.total_runs_24h > 0) THEN round((((stats.successful_runs_24h)::numeric / (stats.total_runs_24h)::numeric) * (100)::numeric), 1)
            ELSE NULL::numeric
        END AS success_rate_24h,
    stats.avg_duration_24h
   FROM ((public.automations a
     LEFT JOIN LATERAL ( SELECT automation_executions.id,
            automation_executions.automation_id,
            automation_executions.celery_task_id,
            automation_executions.started_at,
            automation_executions.completed_at,
            automation_executions.duration_seconds,
            automation_executions.status,
            automation_executions.result,
            automation_executions.error_message,
            automation_executions.error_traceback,
            automation_executions.records_processed,
            automation_executions.triggered_by,
            automation_executions.worker_hostname
           FROM public.automation_executions
          WHERE (automation_executions.automation_id = a.id)
          ORDER BY automation_executions.started_at DESC
         LIMIT 1) latest ON (true))
     LEFT JOIN LATERAL ( SELECT count(*) AS total_runs_24h,
            count(*) FILTER (WHERE ((automation_executions.status)::text = 'success'::text)) AS successful_runs_24h,
            count(*) FILTER (WHERE ((automation_executions.status)::text = 'failed'::text)) AS failed_runs_24h,
            round(avg(automation_executions.duration_seconds), 2) AS avg_duration_24h
           FROM public.automation_executions
          WHERE ((automation_executions.automation_id = a.id) AND (automation_executions.started_at > (now() - '24:00:00'::interval)))) stats ON (true));


ALTER TABLE public.v_automation_status OWNER TO postgres;

--
-- Name: VIEW v_automation_status; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_automation_status IS 'Combined view of automation definitions with latest execution status and 24h statistics';


--
-- Name: v_ava_active_alerts; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_ava_active_alerts AS
 SELECT ava_alerts.category,
    ava_alerts.priority,
    count(*) AS count,
    min(ava_alerts.created_at) AS oldest,
    max(ava_alerts.created_at) AS newest
   FROM public.ava_alerts
  WHERE ((ava_alerts.is_active = true) AND ((ava_alerts.expires_at IS NULL) OR (ava_alerts.expires_at > now())))
  GROUP BY ava_alerts.category, ava_alerts.priority
  ORDER BY ava_alerts.priority, ava_alerts.category;


ALTER TABLE public.v_ava_active_alerts OWNER TO adam;

--
-- Name: VIEW v_ava_active_alerts; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON VIEW public.v_ava_active_alerts IS 'Summary of active alerts by category and priority';


--
-- Name: v_ava_category_summary; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_ava_category_summary AS
 SELECT fs.category,
    count(*) AS feature_count,
    round(avg(er.overall_rating), 2) AS avg_efficiency,
    round(min(er.overall_rating), 2) AS min_efficiency,
    round(max(er.overall_rating), 2) AS max_efficiency,
    sum(COALESCE(ic.open_issues, (0)::bigint)) AS total_open_issues,
    sum(COALESCE(ec.pending, (0)::bigint)) AS total_pending_enhancements
   FROM (((public.ava_feature_specs fs
     LEFT JOIN public.ava_spec_efficiency_ratings er ON ((fs.id = er.spec_id)))
     LEFT JOIN LATERAL ( SELECT count(*) AS open_issues
           FROM public.ava_spec_known_issues
          WHERE ((ava_spec_known_issues.spec_id = fs.id) AND ((ava_spec_known_issues.status)::text <> 'resolved'::text))) ic ON (true))
     LEFT JOIN LATERAL ( SELECT count(*) AS pending
           FROM public.ava_spec_enhancements
          WHERE ((ava_spec_enhancements.spec_id = fs.id) AND ((ava_spec_enhancements.status)::text = ANY ((ARRAY['proposed'::character varying, 'approved'::character varying])::text[])))) ec ON (true))
  WHERE ((fs.is_current = true) AND ((fs.status)::text = 'active'::text))
  GROUP BY fs.category
  ORDER BY (count(*)) DESC;


ALTER TABLE public.v_ava_category_summary OWNER TO postgres;

--
-- Name: VIEW v_ava_category_summary; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_ava_category_summary IS 'Summary statistics by category';


--
-- Name: v_ava_feature_overview; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_ava_feature_overview AS
 SELECT fs.id,
    fs.feature_id,
    fs.feature_name,
    fs.category,
    fs.subcategory,
    fs.status,
    fs.maturity_level,
    fs.purpose,
    er.overall_rating,
    er.code_completeness,
    er.test_coverage,
    er.performance AS perf_rating,
    er.error_handling,
    er.documentation,
    er.priority_level,
    COALESCE(issue_counts.open_issues, (0)::bigint) AS open_issues,
    COALESCE(issue_counts.critical_issues, (0)::bigint) AS critical_issues,
    COALESCE(enhancement_counts.pending_enhancements, (0)::bigint) AS pending_enhancements,
    COALESCE(dep_counts.dependencies, (0)::bigint) AS dependency_count,
    COALESCE(dep_counts.dependents, (0)::bigint) AS dependent_count,
    fs.updated_at,
    fs.analyzed_at
   FROM ((((public.ava_feature_specs fs
     LEFT JOIN public.ava_spec_efficiency_ratings er ON ((fs.id = er.spec_id)))
     LEFT JOIN LATERAL ( SELECT count(*) FILTER (WHERE ((ava_spec_known_issues.status)::text <> 'resolved'::text)) AS open_issues,
            count(*) FILTER (WHERE ((ava_spec_known_issues.severity = 'critical'::public.issue_severity) AND ((ava_spec_known_issues.status)::text <> 'resolved'::text))) AS critical_issues
           FROM public.ava_spec_known_issues
          WHERE (ava_spec_known_issues.spec_id = fs.id)) issue_counts ON (true))
     LEFT JOIN LATERAL ( SELECT count(*) AS pending_enhancements
           FROM public.ava_spec_enhancements
          WHERE ((ava_spec_enhancements.spec_id = fs.id) AND ((ava_spec_enhancements.status)::text = ANY ((ARRAY['proposed'::character varying, 'approved'::character varying])::text[])))) enhancement_counts ON (true))
     LEFT JOIN LATERAL ( SELECT count(*) FILTER (WHERE (ava_spec_dependencies.source_spec_id = fs.id)) AS dependencies,
            count(*) FILTER (WHERE (ava_spec_dependencies.target_spec_id = fs.id)) AS dependents
           FROM public.ava_spec_dependencies
          WHERE ((ava_spec_dependencies.source_spec_id = fs.id) OR (ava_spec_dependencies.target_spec_id = fs.id))) dep_counts ON (true))
  WHERE (fs.is_current = true);


ALTER TABLE public.v_ava_feature_overview OWNER TO postgres;

--
-- Name: VIEW v_ava_feature_overview; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_ava_feature_overview IS 'Comprehensive feature overview with ratings and counts';


--
-- Name: v_ava_features_needing_attention; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_ava_features_needing_attention AS
 SELECT fs.feature_id,
    fs.feature_name,
    fs.category,
    er.overall_rating,
    er.priority_level,
    er.weaknesses,
    COALESCE(ic.critical_count, (0)::bigint) AS critical_issues,
    COALESCE(ic.high_count, (0)::bigint) AS high_issues,
    ARRAY( SELECT e.enhancement_title
           FROM public.ava_spec_enhancements e
          WHERE ((e.spec_id = fs.id) AND (e.priority = ANY (ARRAY['p0_critical'::public.enhancement_priority, 'p1_high'::public.enhancement_priority])) AND ((e.status)::text = ANY ((ARRAY['proposed'::character varying, 'approved'::character varying])::text[])))
         LIMIT 3) AS top_enhancements
   FROM ((public.ava_feature_specs fs
     JOIN public.ava_spec_efficiency_ratings er ON ((fs.id = er.spec_id)))
     LEFT JOIN LATERAL ( SELECT count(*) FILTER (WHERE (ava_spec_known_issues.severity = 'critical'::public.issue_severity)) AS critical_count,
            count(*) FILTER (WHERE (ava_spec_known_issues.severity = 'high'::public.issue_severity)) AS high_count
           FROM public.ava_spec_known_issues
          WHERE ((ava_spec_known_issues.spec_id = fs.id) AND ((ava_spec_known_issues.status)::text <> 'resolved'::text))) ic ON (true))
  WHERE ((fs.is_current = true) AND ((fs.status)::text = 'active'::text) AND ((er.overall_rating < 7.0) OR (ic.critical_count > 0) OR (ic.high_count > 0)))
  ORDER BY er.overall_rating, ic.critical_count DESC;


ALTER TABLE public.v_ava_features_needing_attention OWNER TO postgres;

--
-- Name: VIEW v_ava_features_needing_attention; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_ava_features_needing_attention IS 'Features with low efficiency or critical issues';


--
-- Name: v_ava_goal_dashboard; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_ava_goal_dashboard AS
 SELECT g.id,
    g.goal_name,
    g.goal_type,
    g.target_value,
    g.target_unit,
    g.current_value,
    g.progress_pct,
    g.period_type,
    g.status,
        CASE
            WHEN (g.progress_pct >= (100)::numeric) THEN 'exceeded'::text
            WHEN (g.progress_pct >= (75)::numeric) THEN 'on_track'::text
            WHEN (g.progress_pct >= (50)::numeric) THEN 'moderate'::text
            ELSE 'behind'::text
        END AS progress_status,
    g.updated_at AS last_updated
   FROM public.ava_user_goals g
  WHERE ((g.status)::text = 'active'::text)
  ORDER BY g.progress_pct DESC;


ALTER TABLE public.v_ava_goal_dashboard OWNER TO adam;

--
-- Name: VIEW v_ava_goal_dashboard; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON VIEW public.v_ava_goal_dashboard IS 'Dashboard view of active goals with progress status';


--
-- Name: v_ava_integration_usage; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_ava_integration_usage AS
 SELECT si.integration_name,
    si.integration_type,
    count(DISTINCT fs.id) AS feature_count,
    count(*) FILTER (WHERE si.is_critical) AS critical_usage_count,
    array_agg(DISTINCT fs.category) AS categories_using,
    array_agg(DISTINCT fs.feature_name ORDER BY fs.feature_name) AS features_using
   FROM (public.ava_spec_integrations si
     JOIN public.ava_feature_specs fs ON ((si.spec_id = fs.id)))
  WHERE ((fs.is_current = true) AND ((fs.status)::text = 'active'::text))
  GROUP BY si.integration_name, si.integration_type
  ORDER BY (count(DISTINCT fs.id)) DESC;


ALTER TABLE public.v_ava_integration_usage OWNER TO postgres;

--
-- Name: VIEW v_ava_integration_usage; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_ava_integration_usage IS 'Summary of integration usage across features';


--
-- Name: v_ava_pattern_summary; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_ava_pattern_summary AS
 SELECT ava_learning_patterns.user_id,
    ava_learning_patterns.pattern_type,
    count(*) AS pattern_count,
    avg(ava_learning_patterns.win_rate) AS avg_win_rate,
    avg(ava_learning_patterns.confidence_score) AS avg_confidence,
    sum(ava_learning_patterns.sample_count) AS total_samples
   FROM public.ava_learning_patterns
  WHERE (ava_learning_patterns.active = true)
  GROUP BY ava_learning_patterns.user_id, ava_learning_patterns.pattern_type
  ORDER BY (avg(ava_learning_patterns.win_rate)) DESC;


ALTER TABLE public.v_ava_pattern_summary OWNER TO adam;

--
-- Name: VIEW v_ava_pattern_summary; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON VIEW public.v_ava_pattern_summary IS 'Summary of learned patterns by type';


--
-- Name: v_earnings_beat_stats; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_earnings_beat_stats AS
 SELECT earnings_history.symbol,
    count(*) AS total_reports,
    count(*) FILTER (WHERE ((earnings_history.beat_miss)::text = 'beat'::text)) AS beats,
    count(*) FILTER (WHERE ((earnings_history.beat_miss)::text = 'miss'::text)) AS misses,
    count(*) FILTER (WHERE ((earnings_history.beat_miss)::text = 'meet'::text)) AS meets,
    round(((100.0 * (count(*) FILTER (WHERE ((earnings_history.beat_miss)::text = 'beat'::text)))::numeric) / (NULLIF(count(*), 0))::numeric), 2) AS beat_rate_pct,
    round(avg(earnings_history.eps_surprise_percent), 2) AS avg_surprise_pct,
    round(stddev(earnings_history.eps_surprise_percent), 2) AS surprise_volatility,
    max(earnings_history.report_date) AS last_earnings_date,
    min(earnings_history.report_date) AS first_earnings_date
   FROM public.earnings_history
  WHERE (earnings_history.report_date >= (CURRENT_DATE - '2 years'::interval))
  GROUP BY earnings_history.symbol
  ORDER BY (round(((100.0 * (count(*) FILTER (WHERE ((earnings_history.beat_miss)::text = 'beat'::text)))::numeric) / (NULLIF(count(*), 0))::numeric), 2)) DESC;


ALTER TABLE public.v_earnings_beat_stats OWNER TO adam;

--
-- Name: v_high_conviction_earnings; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_high_conviction_earnings AS
 SELECT ee.symbol,
    ee.earnings_date,
    ee.earnings_time,
    ee.eps_estimate,
    ee.pre_earnings_iv,
    stats.beat_rate_pct,
    stats.avg_surprise_pct,
    stats.total_reports,
        CASE
            WHEN (stats.beat_rate_pct >= (75)::numeric) THEN 'Strong Beat History'::text
            WHEN (stats.beat_rate_pct <= (25)::numeric) THEN 'Strong Miss History'::text
            ELSE 'Mixed History'::text
        END AS conviction_signal
   FROM (public.earnings_events ee
     JOIN public.v_earnings_beat_stats stats ON (((ee.symbol)::text = (stats.symbol)::text)))
  WHERE ((ee.has_occurred = false) AND (ee.earnings_date >= CURRENT_DATE) AND (ee.earnings_date <= (CURRENT_DATE + '14 days'::interval)) AND (stats.total_reports >= 4) AND ((stats.beat_rate_pct >= (75)::numeric) OR (stats.beat_rate_pct <= (25)::numeric)))
  ORDER BY ee.earnings_date, stats.beat_rate_pct DESC;


ALTER TABLE public.v_high_conviction_earnings OWNER TO adam;

--
-- Name: v_kalshi_college_active; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_kalshi_college_active AS
 SELECT m.ticker,
    m.title,
    m.home_team,
    m.away_team,
    m.game_date,
    m.yes_price,
    m.no_price,
    m.volume,
    m.close_time,
    p.predicted_outcome,
    p.confidence_score,
    p.edge_percentage,
    p.overall_rank,
    p.recommended_action,
    p.recommended_stake_pct,
    p.reasoning
   FROM (public.kalshi_markets m
     LEFT JOIN public.kalshi_predictions p ON ((m.id = p.market_id)))
  WHERE (((m.market_type)::text = 'college'::text) AND ((m.status)::text = 'open'::text))
  ORDER BY p.overall_rank;


ALTER TABLE public.v_kalshi_college_active OWNER TO postgres;

--
-- Name: VIEW v_kalshi_college_active; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_kalshi_college_active IS 'Active College Football markets with AI predictions, ranked by opportunity';


--
-- Name: v_kalshi_nfl_active; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_kalshi_nfl_active AS
 SELECT m.ticker,
    m.title,
    m.home_team,
    m.away_team,
    m.game_date,
    m.yes_price,
    m.no_price,
    m.volume,
    m.close_time,
    p.predicted_outcome,
    p.confidence_score,
    p.edge_percentage,
    p.overall_rank,
    p.recommended_action,
    p.recommended_stake_pct,
    p.reasoning
   FROM (public.kalshi_markets m
     LEFT JOIN public.kalshi_predictions p ON ((m.id = p.market_id)))
  WHERE (((m.market_type)::text = 'nfl'::text) AND ((m.status)::text = 'open'::text))
  ORDER BY p.overall_rank;


ALTER TABLE public.v_kalshi_nfl_active OWNER TO postgres;

--
-- Name: VIEW v_kalshi_nfl_active; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_kalshi_nfl_active IS 'Active NFL markets with AI predictions, ranked by opportunity';


--
-- Name: v_kalshi_top_opportunities; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_kalshi_top_opportunities AS
 SELECT m.market_type,
    m.ticker,
    m.title,
    m.home_team,
    m.away_team,
    m.game_date,
    m.yes_price,
    m.no_price,
    m.volume,
    p.predicted_outcome,
    p.confidence_score,
    p.edge_percentage,
    p.overall_rank,
    p.recommended_action,
    p.recommended_stake_pct,
    p.reasoning,
    p.key_factors
   FROM (public.kalshi_markets m
     JOIN public.kalshi_predictions p ON ((m.id = p.market_id)))
  WHERE (((m.status)::text = ANY ((ARRAY['open'::character varying, 'active'::character varying])::text[])) AND ((p.recommended_action)::text = ANY ((ARRAY['strong_buy'::character varying, 'buy'::character varying])::text[])) AND (p.edge_percentage > (0)::numeric))
  ORDER BY p.overall_rank
 LIMIT 50;


ALTER TABLE public.v_kalshi_top_opportunities OWNER TO postgres;

--
-- Name: VIEW v_kalshi_top_opportunities; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_kalshi_top_opportunities IS 'Top 50 betting opportunities ranked by AI analysis';


--
-- Name: v_nfl_kalshi_opportunities; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_nfl_kalshi_opportunities AS
 SELECT g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.quarter,
    g.time_remaining,
    km.ticker,
    km.title,
    km.yes_price,
    km.volume,
    kp.predicted_outcome,
    kp.confidence_score,
    kp.edge_percentage,
    kp.recommended_action,
        CASE
            WHEN ((g.quarter >= 4) AND (g.home_score > (g.away_score + 7))) THEN 'home_lock'::text
            WHEN ((g.quarter >= 4) AND (g.away_score > (g.home_score + 7))) THEN 'away_lock'::text
            WHEN (g.quarter <= 2) THEN 'early'::text
            ELSE 'competitive'::text
        END AS game_state,
    g.last_updated AS game_last_updated,
    km.last_updated AS market_last_updated
   FROM ((public.nfl_games g
     JOIN public.kalshi_markets km ON ((((km.home_team)::text = (g.home_team)::text) OR ((km.away_team)::text = (g.away_team)::text))))
     LEFT JOIN public.kalshi_predictions kp ON ((km.id = kp.market_id)))
  WHERE ((g.is_live = true) AND ((km.status)::text = 'open'::text) AND (kp.edge_percentage > (5)::numeric))
  ORDER BY kp.edge_percentage DESC NULLS LAST;


ALTER TABLE public.v_nfl_kalshi_opportunities OWNER TO postgres;

--
-- Name: VIEW v_nfl_kalshi_opportunities; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_nfl_kalshi_opportunities IS 'Live NFL games with high-value Kalshi betting opportunities';


--
-- Name: v_nfl_live_games; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_nfl_live_games AS
 SELECT g.id,
    g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.quarter,
    g.time_remaining,
    g.possession,
    g.spread_home,
    g.over_under,
    g.venue,
    g.temperature,
    g.weather_condition,
    g.last_updated,
    (g.home_score - g.away_score) AS score_diff,
    ( SELECT count(*) AS count
           FROM public.kalshi_markets km
          WHERE ((((km.home_team)::text = (g.home_team)::text) OR ((km.away_team)::text = (g.away_team)::text)) AND ((km.status)::text = 'open'::text))) AS active_markets_count
   FROM public.nfl_games g
  WHERE (g.is_live = true)
  ORDER BY g.game_time;


ALTER TABLE public.v_nfl_live_games OWNER TO postgres;

--
-- Name: VIEW v_nfl_live_games; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_nfl_live_games IS 'Currently live NFL games with scores and market data';


--
-- Name: v_nfl_prediction_accuracy; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_nfl_prediction_accuracy AS
 SELECT g.id,
    g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.spread_home AS predicted_spread,
    (g.home_score - g.away_score) AS actual_spread,
    abs((g.spread_home - ((g.home_score - g.away_score))::numeric)) AS spread_error,
    g.over_under AS predicted_total,
    (g.home_score + g.away_score) AS actual_total,
    abs((g.over_under - ((g.home_score + g.away_score))::numeric)) AS total_error,
    g.game_status,
    g.finished_at
   FROM public.nfl_games g
  WHERE (((g.game_status)::text = 'final'::text) AND (g.spread_home IS NOT NULL))
  ORDER BY g.finished_at DESC;


ALTER TABLE public.v_nfl_prediction_accuracy OWNER TO postgres;

--
-- Name: VIEW v_nfl_prediction_accuracy; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_nfl_prediction_accuracy IS 'Compare betting lines to actual game results';


--
-- Name: v_nfl_significant_plays; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_nfl_significant_plays AS
 SELECT p.id,
    p.game_id,
    g.home_team,
    g.away_team,
    p.quarter,
    p.time_remaining,
    p.play_type,
    p.description,
    p.yards_gained,
    p.is_scoring_play,
    p.is_turnover,
    p.offense_team,
    p.player_name,
    p.points_home,
    p.points_away,
    p.created_at
   FROM (public.nfl_plays p
     JOIN public.nfl_games g ON ((p.game_id = g.id)))
  WHERE (((p.is_scoring_play = true) OR (p.is_turnover = true) OR (abs(p.yards_gained) >= 20)) AND (p.created_at > (now() - '02:00:00'::interval)))
  ORDER BY p.created_at DESC
 LIMIT 100;


ALTER TABLE public.v_nfl_significant_plays OWNER TO postgres;

--
-- Name: VIEW v_nfl_significant_plays; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON VIEW public.v_nfl_significant_plays IS 'Recent high-impact plays across all games';


--
-- Name: v_upcoming_earnings; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.v_upcoming_earnings AS
 SELECT ee.symbol,
    ee.earnings_date,
    ee.earnings_time,
    ee.eps_estimate,
    ee.revenue_estimate,
    ee.whisper_number,
    ee.pre_earnings_iv,
    ee.call_datetime,
    ee.is_confirmed,
    ( SELECT round(((100.0 * (count(*) FILTER (WHERE ((eh.beat_miss)::text = 'beat'::text)))::numeric) / (NULLIF(count(*), 0))::numeric), 2) AS round
           FROM public.earnings_history eh
          WHERE (((eh.symbol)::text = (ee.symbol)::text) AND (eh.report_date >= (CURRENT_DATE - '2 years'::interval)))) AS historical_beat_rate_pct,
    ( SELECT eh.eps_surprise_percent
           FROM public.earnings_history eh
          WHERE ((eh.symbol)::text = (ee.symbol)::text)
          ORDER BY eh.report_date DESC
         LIMIT 1) AS last_quarter_surprise_pct
   FROM public.earnings_events ee
  WHERE ((ee.has_occurred = false) AND (ee.earnings_date >= CURRENT_DATE) AND (ee.earnings_date <= (CURRENT_DATE + '30 days'::interval)))
  ORDER BY ee.earnings_date, ee.symbol;


ALTER TABLE public.v_upcoming_earnings OWNER TO adam;

--
-- Name: xtrades_profiles; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.xtrades_profiles (
    id integer NOT NULL,
    username character varying(255) NOT NULL,
    display_name character varying(255),
    active boolean DEFAULT true,
    added_date timestamp with time zone DEFAULT now(),
    last_sync timestamp with time zone,
    last_sync_status character varying(50),
    total_trades_scraped integer DEFAULT 0,
    notes text,
    CONSTRAINT chk_sync_status CHECK (((last_sync_status)::text = ANY ((ARRAY['success'::character varying, 'error'::character varying, 'pending'::character varying, NULL::character varying])::text[])))
);


ALTER TABLE public.xtrades_profiles OWNER TO adam;

--
-- Name: TABLE xtrades_profiles; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.xtrades_profiles IS 'Stores Xtrades.net profiles to monitor for trading alerts';


--
-- Name: COLUMN xtrades_profiles.username; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_profiles.username IS 'Unique Xtrades.net username';


--
-- Name: COLUMN xtrades_profiles.active; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_profiles.active IS 'Whether this profile is actively being monitored';


--
-- Name: COLUMN xtrades_profiles.last_sync; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_profiles.last_sync IS 'Timestamp of the last successful data sync';


--
-- Name: COLUMN xtrades_profiles.total_trades_scraped; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_profiles.total_trades_scraped IS 'Running count of trades collected from this profile';


--
-- Name: xtrades_trades; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.xtrades_trades (
    id integer NOT NULL,
    profile_id integer NOT NULL,
    ticker character varying(20) NOT NULL,
    strategy character varying(100),
    action character varying(20),
    entry_price numeric(10,2),
    entry_date timestamp with time zone,
    exit_price numeric(10,2),
    exit_date timestamp with time zone,
    quantity integer,
    pnl numeric(10,2),
    pnl_percent numeric(10,2),
    status character varying(20) DEFAULT 'open'::character varying,
    strike_price numeric(10,2),
    expiration_date date,
    alert_text text,
    alert_timestamp timestamp with time zone NOT NULL,
    scraped_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    xtrades_alert_id character varying(255),
    CONSTRAINT chk_trade_action CHECK (((action)::text = ANY ((ARRAY['BTO'::character varying, 'STC'::character varying, 'BTC'::character varying, 'STO'::character varying, 'OPEN'::character varying, 'CLOSE'::character varying, NULL::character varying])::text[]))),
    CONSTRAINT chk_trade_status CHECK (((status)::text = ANY ((ARRAY['open'::character varying, 'closed'::character varying, 'expired'::character varying])::text[])))
);


ALTER TABLE public.xtrades_trades OWNER TO adam;

--
-- Name: TABLE xtrades_trades; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.xtrades_trades IS 'Individual trades and positions scraped from Xtrades profiles';


--
-- Name: COLUMN xtrades_trades.ticker; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.ticker IS 'Stock ticker symbol';


--
-- Name: COLUMN xtrades_trades.strategy; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.strategy IS 'Options strategy type (CSP, CC, spreads, etc.)';


--
-- Name: COLUMN xtrades_trades.action; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.action IS 'Trade action: BTO=Buy To Open, STC=Sell To Close, BTC=Buy To Close, STO=Sell To Open';


--
-- Name: COLUMN xtrades_trades.pnl; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.pnl IS 'Profit/Loss in dollars';


--
-- Name: COLUMN xtrades_trades.pnl_percent; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.pnl_percent IS 'Profit/Loss as percentage';


--
-- Name: COLUMN xtrades_trades.alert_text; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.alert_text IS 'Raw alert text from Xtrades.net';


--
-- Name: COLUMN xtrades_trades.xtrades_alert_id; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_trades.xtrades_alert_id IS 'Unique identifier from Xtrades to prevent duplicates';


--
-- Name: xtrades_alerts; Type: VIEW; Schema: public; Owner: adam
--

CREATE VIEW public.xtrades_alerts AS
 SELECT t.id,
    t.ticker AS symbol,
    t.action,
    p.display_name AS trader_name,
    t.scraped_at AS created_at,
    t.strike_price,
    t.expiration_date
   FROM (public.xtrades_trades t
     JOIN public.xtrades_profiles p ON ((t.profile_id = p.id)));


ALTER TABLE public.xtrades_alerts OWNER TO adam;

--
-- Name: xtrades_notifications; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.xtrades_notifications (
    id integer NOT NULL,
    trade_id integer NOT NULL,
    notification_type character varying(50) NOT NULL,
    sent_at timestamp with time zone DEFAULT now(),
    telegram_message_id character varying(255),
    status character varying(20) DEFAULT 'sent'::character varying,
    error_message text,
    CONSTRAINT chk_notification_status CHECK (((status)::text = ANY ((ARRAY['sent'::character varying, 'failed'::character varying])::text[]))),
    CONSTRAINT chk_notification_type CHECK (((notification_type)::text = ANY ((ARRAY['new_trade'::character varying, 'trade_update'::character varying, 'trade_closed'::character varying])::text[])))
);


ALTER TABLE public.xtrades_notifications OWNER TO adam;

--
-- Name: TABLE xtrades_notifications; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.xtrades_notifications IS 'Tracks notifications sent for trades to prevent duplicates';


--
-- Name: COLUMN xtrades_notifications.notification_type; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_notifications.notification_type IS 'Type of notification: new_trade, trade_update, or trade_closed';


--
-- Name: COLUMN xtrades_notifications.telegram_message_id; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_notifications.telegram_message_id IS 'Telegram message ID for tracking and potential editing';


--
-- Name: xtrades_notifications_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.xtrades_notifications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.xtrades_notifications_id_seq OWNER TO adam;

--
-- Name: xtrades_notifications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.xtrades_notifications_id_seq OWNED BY public.xtrades_notifications.id;


--
-- Name: xtrades_profiles_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.xtrades_profiles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.xtrades_profiles_id_seq OWNER TO adam;

--
-- Name: xtrades_profiles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.xtrades_profiles_id_seq OWNED BY public.xtrades_profiles.id;


--
-- Name: xtrades_sync_log; Type: TABLE; Schema: public; Owner: adam
--

CREATE TABLE public.xtrades_sync_log (
    id integer NOT NULL,
    sync_timestamp timestamp with time zone DEFAULT now(),
    profiles_synced integer DEFAULT 0,
    trades_found integer DEFAULT 0,
    new_trades integer DEFAULT 0,
    updated_trades integer DEFAULT 0,
    errors text,
    duration_seconds numeric(10,2),
    status character varying(50) DEFAULT 'success'::character varying,
    CONSTRAINT chk_sync_log_status CHECK (((status)::text = ANY ((ARRAY['success'::character varying, 'partial'::character varying, 'failed'::character varying])::text[])))
);


ALTER TABLE public.xtrades_sync_log OWNER TO adam;

--
-- Name: TABLE xtrades_sync_log; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON TABLE public.xtrades_sync_log IS 'Audit log of all synchronization operations';


--
-- Name: COLUMN xtrades_sync_log.profiles_synced; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_sync_log.profiles_synced IS 'Number of profiles processed in this sync';


--
-- Name: COLUMN xtrades_sync_log.new_trades; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_sync_log.new_trades IS 'Number of new trades discovered';


--
-- Name: COLUMN xtrades_sync_log.updated_trades; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_sync_log.updated_trades IS 'Number of existing trades updated';


--
-- Name: COLUMN xtrades_sync_log.duration_seconds; Type: COMMENT; Schema: public; Owner: adam
--

COMMENT ON COLUMN public.xtrades_sync_log.duration_seconds IS 'How long the sync operation took';


--
-- Name: xtrades_sync_log_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.xtrades_sync_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.xtrades_sync_log_id_seq OWNER TO adam;

--
-- Name: xtrades_sync_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.xtrades_sync_log_id_seq OWNED BY public.xtrades_sync_log.id;


--
-- Name: xtrades_trades_id_seq; Type: SEQUENCE; Schema: public; Owner: adam
--

CREATE SEQUENCE public.xtrades_trades_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.xtrades_trades_id_seq OWNER TO adam;

--
-- Name: xtrades_trades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: adam
--

ALTER SEQUENCE public.xtrades_trades_id_seq OWNED BY public.xtrades_trades.id;


--
-- Name: agent_execution_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_execution_log ALTER COLUMN id SET DEFAULT nextval('public.agent_execution_log_id_seq'::regclass);


--
-- Name: agent_feedback id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_feedback ALTER COLUMN id SET DEFAULT nextval('public.agent_feedback_id_seq'::regclass);


--
-- Name: agent_memory id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_memory ALTER COLUMN id SET DEFAULT nextval('public.agent_memory_id_seq'::regclass);


--
-- Name: ai_betting_recommendations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ai_betting_recommendations ALTER COLUMN id SET DEFAULT nextval('public.ai_betting_recommendations_id_seq'::regclass);


--
-- Name: automation_executions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_executions ALTER COLUMN id SET DEFAULT nextval('public.automation_executions_id_seq'::regclass);


--
-- Name: automation_state_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_state_log ALTER COLUMN id SET DEFAULT nextval('public.automation_state_log_id_seq'::regclass);


--
-- Name: automations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automations ALTER COLUMN id SET DEFAULT nextval('public.automations_id_seq'::regclass);


--
-- Name: ava_alert_deliveries id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_deliveries ALTER COLUMN id SET DEFAULT nextval('public.ava_alert_deliveries_id_seq'::regclass);


--
-- Name: ava_alert_preferences id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_preferences ALTER COLUMN id SET DEFAULT nextval('public.ava_alert_preferences_id_seq'::regclass);


--
-- Name: ava_alert_rate_limits id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_rate_limits ALTER COLUMN id SET DEFAULT nextval('public.ava_alert_rate_limits_id_seq'::regclass);


--
-- Name: ava_alerts id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alerts ALTER COLUMN id SET DEFAULT nextval('public.ava_alerts_id_seq'::regclass);


--
-- Name: ava_chat_recommendations id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_chat_recommendations ALTER COLUMN id SET DEFAULT nextval('public.ava_chat_recommendations_id_seq'::regclass);


--
-- Name: ava_feature_specs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_feature_specs ALTER COLUMN id SET DEFAULT nextval('public.ava_feature_specs_id_seq'::regclass);


--
-- Name: ava_generated_reports id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_generated_reports ALTER COLUMN id SET DEFAULT nextval('public.ava_generated_reports_id_seq'::regclass);


--
-- Name: ava_goal_progress_history id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_goal_progress_history ALTER COLUMN id SET DEFAULT nextval('public.ava_goal_progress_history_id_seq'::regclass);


--
-- Name: ava_iv_history id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_iv_history ALTER COLUMN id SET DEFAULT nextval('public.ava_iv_history_id_seq'::regclass);


--
-- Name: ava_learning_patterns id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_learning_patterns ALTER COLUMN id SET DEFAULT nextval('public.ava_learning_patterns_id_seq'::regclass);


--
-- Name: ava_opportunity_scans id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_opportunity_scans ALTER COLUMN id SET DEFAULT nextval('public.ava_opportunity_scans_id_seq'::regclass);


--
-- Name: ava_spec_api_endpoints id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_api_endpoints ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_api_endpoints_id_seq'::regclass);


--
-- Name: ava_spec_database_tables id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_database_tables ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_database_tables_id_seq'::regclass);


--
-- Name: ava_spec_dependencies id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_dependencies ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_dependencies_id_seq'::regclass);


--
-- Name: ava_spec_efficiency_ratings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_efficiency_ratings ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_efficiency_ratings_id_seq'::regclass);


--
-- Name: ava_spec_enhancements id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_enhancements ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_enhancements_id_seq'::regclass);


--
-- Name: ava_spec_error_handling id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_error_handling ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_error_handling_id_seq'::regclass);


--
-- Name: ava_spec_integrations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_integrations ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_integrations_id_seq'::regclass);


--
-- Name: ava_spec_known_issues id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_known_issues ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_known_issues_id_seq'::regclass);


--
-- Name: ava_spec_performance_metrics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_performance_metrics ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_performance_metrics_id_seq'::regclass);


--
-- Name: ava_spec_source_files id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_source_files ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_source_files_id_seq'::regclass);


--
-- Name: ava_spec_tags id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_tags ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_tags_id_seq'::regclass);


--
-- Name: ava_spec_version_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_version_history ALTER COLUMN id SET DEFAULT nextval('public.ava_spec_version_history_id_seq'::regclass);


--
-- Name: ava_user_goals id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_user_goals ALTER COLUMN id SET DEFAULT nextval('public.ava_user_goals_id_seq'::regclass);


--
-- Name: earnings_alerts id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_alerts ALTER COLUMN id SET DEFAULT nextval('public.earnings_alerts_id_seq'::regclass);


--
-- Name: earnings_events id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_events ALTER COLUMN id SET DEFAULT nextval('public.earnings_events_id_seq'::regclass);


--
-- Name: earnings_history id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_history ALTER COLUMN id SET DEFAULT nextval('public.earnings_history_id_seq'::regclass);


--
-- Name: earnings_sync_status id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_sync_status ALTER COLUMN id SET DEFAULT nextval('public.earnings_sync_status_id_seq'::regclass);


--
-- Name: etfs_universe id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etfs_universe ALTER COLUMN id SET DEFAULT nextval('public.etfs_universe_id_seq'::regclass);


--
-- Name: kalshi_markets id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_markets ALTER COLUMN id SET DEFAULT nextval('public.kalshi_markets_id_seq'::regclass);


--
-- Name: kalshi_predictions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_predictions ALTER COLUMN id SET DEFAULT nextval('public.kalshi_predictions_id_seq'::regclass);


--
-- Name: kalshi_price_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_price_history ALTER COLUMN id SET DEFAULT nextval('public.kalshi_price_history_id_seq'::regclass);


--
-- Name: kalshi_sync_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_sync_log ALTER COLUMN id SET DEFAULT nextval('public.kalshi_sync_log_id_seq'::regclass);


--
-- Name: live_odds_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_odds_snapshots ALTER COLUMN id SET DEFAULT nextval('public.live_odds_snapshots_id_seq'::regclass);


--
-- Name: live_prediction_snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_prediction_snapshots ALTER COLUMN id SET DEFAULT nextval('public.live_prediction_snapshots_id_seq'::regclass);


--
-- Name: model_performance id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_performance ALTER COLUMN id SET DEFAULT nextval('public.model_performance_id_seq'::regclass);


--
-- Name: nba_games id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nba_games ALTER COLUMN id SET DEFAULT nextval('public.nba_games_id_seq'::regclass);


--
-- Name: ncaa_basketball_games id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_basketball_games ALTER COLUMN id SET DEFAULT nextval('public.ncaa_basketball_games_id_seq'::regclass);


--
-- Name: ncaa_football_games id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_football_games ALTER COLUMN id SET DEFAULT nextval('public.ncaa_football_games_id_seq'::regclass);


--
-- Name: nfl_alert_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history ALTER COLUMN id SET DEFAULT nextval('public.nfl_alert_history_id_seq'::regclass);


--
-- Name: nfl_alert_triggers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_triggers ALTER COLUMN id SET DEFAULT nextval('public.nfl_alert_triggers_id_seq'::regclass);


--
-- Name: nfl_data_sync_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_data_sync_log ALTER COLUMN id SET DEFAULT nextval('public.nfl_data_sync_log_id_seq'::regclass);


--
-- Name: nfl_games id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_games ALTER COLUMN id SET DEFAULT nextval('public.nfl_games_id_seq'::regclass);


--
-- Name: nfl_injuries id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_injuries ALTER COLUMN id SET DEFAULT nextval('public.nfl_injuries_id_seq'::regclass);


--
-- Name: nfl_kalshi_correlations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_kalshi_correlations ALTER COLUMN id SET DEFAULT nextval('public.nfl_kalshi_correlations_id_seq'::regclass);


--
-- Name: nfl_player_stats id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_player_stats ALTER COLUMN id SET DEFAULT nextval('public.nfl_player_stats_id_seq'::regclass);


--
-- Name: nfl_plays id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_plays ALTER COLUMN id SET DEFAULT nextval('public.nfl_plays_id_seq'::regclass);


--
-- Name: nfl_social_sentiment id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_social_sentiment ALTER COLUMN id SET DEFAULT nextval('public.nfl_social_sentiment_id_seq'::regclass);


--
-- Name: odds_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.odds_history ALTER COLUMN id SET DEFAULT nextval('public.odds_history_id_seq'::regclass);


--
-- Name: portfolio_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.portfolio_history ALTER COLUMN id SET DEFAULT nextval('public.portfolio_history_id_seq'::regclass);


--
-- Name: prediction_results id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prediction_results ALTER COLUMN id SET DEFAULT nextval('public.prediction_results_id_seq'::regclass);


--
-- Name: premium_opportunities id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_opportunities ALTER COLUMN id SET DEFAULT nextval('public.premium_opportunities_id_seq'::regclass);


--
-- Name: premium_scan_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_scan_history ALTER COLUMN id SET DEFAULT nextval('public.premium_scan_history_id_seq'::regclass);


--
-- Name: scanner_results id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.scanner_results ALTER COLUMN id SET DEFAULT nextval('public.scanner_results_id_seq'::regclass);


--
-- Name: scanner_watchlists id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_watchlists ALTER COLUMN id SET DEFAULT nextval('public.scanner_watchlists_id_seq'::regclass);


--
-- Name: scanner_watchlists_sync_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_watchlists_sync_log ALTER COLUMN id SET DEFAULT nextval('public.scanner_watchlists_sync_log_id_seq'::regclass);


--
-- Name: stock_ai_scores id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_ai_scores ALTER COLUMN id SET DEFAULT nextval('public.stock_ai_scores_id_seq'::regclass);


--
-- Name: stocks_universe id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks_universe ALTER COLUMN id SET DEFAULT nextval('public.stocks_universe_id_seq'::regclass);


--
-- Name: trade_history id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.trade_history ALTER COLUMN id SET DEFAULT nextval('public.trade_history_id_seq'::regclass);


--
-- Name: trade_journal id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.trade_journal ALTER COLUMN id SET DEFAULT nextval('public.trade_journal_id_seq'::regclass);


--
-- Name: tv_symbols_api id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_symbols_api ALTER COLUMN id SET DEFAULT nextval('public.tv_symbols_api_id_seq'::regclass);


--
-- Name: tv_watchlists_api id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_watchlists_api ALTER COLUMN id SET DEFAULT nextval('public.tv_watchlists_api_id_seq'::regclass);


--
-- Name: user_bets id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_bets ALTER COLUMN id SET DEFAULT nextval('public.user_bets_id_seq'::regclass);


--
-- Name: xtrades_notifications id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_notifications ALTER COLUMN id SET DEFAULT nextval('public.xtrades_notifications_id_seq'::regclass);


--
-- Name: xtrades_profiles id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_profiles ALTER COLUMN id SET DEFAULT nextval('public.xtrades_profiles_id_seq'::regclass);


--
-- Name: xtrades_sync_log id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_sync_log ALTER COLUMN id SET DEFAULT nextval('public.xtrades_sync_log_id_seq'::regclass);


--
-- Name: xtrades_trades id; Type: DEFAULT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_trades ALTER COLUMN id SET DEFAULT nextval('public.xtrades_trades_id_seq'::regclass);


--
-- Name: agent_execution_log agent_execution_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_execution_log
    ADD CONSTRAINT agent_execution_log_pkey PRIMARY KEY (id);


--
-- Name: agent_feedback agent_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_feedback
    ADD CONSTRAINT agent_feedback_pkey PRIMARY KEY (id);


--
-- Name: agent_memory agent_memory_agent_name_memory_key_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_memory
    ADD CONSTRAINT agent_memory_agent_name_memory_key_key UNIQUE (agent_name, memory_key);


--
-- Name: agent_memory agent_memory_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_memory
    ADD CONSTRAINT agent_memory_pkey PRIMARY KEY (id);


--
-- Name: agent_performance agent_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.agent_performance
    ADD CONSTRAINT agent_performance_pkey PRIMARY KEY (agent_name);


--
-- Name: ai_betting_recommendations ai_betting_recommendations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ai_betting_recommendations
    ADD CONSTRAINT ai_betting_recommendations_pkey PRIMARY KEY (id);


--
-- Name: automation_executions automation_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_executions
    ADD CONSTRAINT automation_executions_pkey PRIMARY KEY (id);


--
-- Name: automation_state_log automation_state_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_state_log
    ADD CONSTRAINT automation_state_log_pkey PRIMARY KEY (id);


--
-- Name: automations automations_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automations
    ADD CONSTRAINT automations_name_key UNIQUE (name);


--
-- Name: automations automations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automations
    ADD CONSTRAINT automations_pkey PRIMARY KEY (id);


--
-- Name: ava_alert_deliveries ava_alert_deliveries_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_deliveries
    ADD CONSTRAINT ava_alert_deliveries_pkey PRIMARY KEY (id);


--
-- Name: ava_alert_preferences ava_alert_preferences_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_preferences
    ADD CONSTRAINT ava_alert_preferences_pkey PRIMARY KEY (id);


--
-- Name: ava_alert_preferences ava_alert_preferences_user_id_platform_category_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_preferences
    ADD CONSTRAINT ava_alert_preferences_user_id_platform_category_key UNIQUE (user_id, platform, category);


--
-- Name: ava_alert_rate_limits ava_alert_rate_limits_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_rate_limits
    ADD CONSTRAINT ava_alert_rate_limits_pkey PRIMARY KEY (id);


--
-- Name: ava_alert_rate_limits ava_alert_rate_limits_user_id_channel_window_start_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_rate_limits
    ADD CONSTRAINT ava_alert_rate_limits_user_id_channel_window_start_key UNIQUE (user_id, channel, window_start);


--
-- Name: ava_alerts ava_alerts_fingerprint_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alerts
    ADD CONSTRAINT ava_alerts_fingerprint_key UNIQUE (fingerprint);


--
-- Name: ava_alerts ava_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alerts
    ADD CONSTRAINT ava_alerts_pkey PRIMARY KEY (id);


--
-- Name: ava_chat_recommendations ava_chat_recommendations_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_chat_recommendations
    ADD CONSTRAINT ava_chat_recommendations_pkey PRIMARY KEY (id);


--
-- Name: ava_feature_specs ava_feature_specs_feature_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_feature_specs
    ADD CONSTRAINT ava_feature_specs_feature_id_key UNIQUE (feature_id);


--
-- Name: ava_feature_specs ava_feature_specs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_feature_specs
    ADD CONSTRAINT ava_feature_specs_pkey PRIMARY KEY (id);


--
-- Name: ava_generated_reports ava_generated_reports_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_generated_reports
    ADD CONSTRAINT ava_generated_reports_pkey PRIMARY KEY (id);


--
-- Name: ava_generated_reports ava_generated_reports_report_type_report_date_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_generated_reports
    ADD CONSTRAINT ava_generated_reports_report_type_report_date_key UNIQUE (report_type, report_date);


--
-- Name: ava_goal_progress_history ava_goal_progress_history_goal_id_snapshot_date_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_goal_progress_history
    ADD CONSTRAINT ava_goal_progress_history_goal_id_snapshot_date_key UNIQUE (goal_id, snapshot_date);


--
-- Name: ava_goal_progress_history ava_goal_progress_history_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_goal_progress_history
    ADD CONSTRAINT ava_goal_progress_history_pkey PRIMARY KEY (id);


--
-- Name: ava_iv_history ava_iv_history_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_iv_history
    ADD CONSTRAINT ava_iv_history_pkey PRIMARY KEY (id);


--
-- Name: ava_iv_history ava_iv_history_symbol_date_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_iv_history
    ADD CONSTRAINT ava_iv_history_symbol_date_key UNIQUE (symbol, date);


--
-- Name: ava_learning_patterns ava_learning_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_learning_patterns
    ADD CONSTRAINT ava_learning_patterns_pkey PRIMARY KEY (id);


--
-- Name: ava_learning_patterns ava_learning_patterns_user_id_platform_pattern_type_pattern_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_learning_patterns
    ADD CONSTRAINT ava_learning_patterns_user_id_platform_pattern_type_pattern_key UNIQUE (user_id, platform, pattern_type, pattern_name);


--
-- Name: ava_opportunity_scans ava_opportunity_scans_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_opportunity_scans
    ADD CONSTRAINT ava_opportunity_scans_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_api_endpoints ava_spec_api_endpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_api_endpoints
    ADD CONSTRAINT ava_spec_api_endpoints_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_api_endpoints ava_spec_api_endpoints_spec_id_method_path_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_api_endpoints
    ADD CONSTRAINT ava_spec_api_endpoints_spec_id_method_path_key UNIQUE (spec_id, method, path);


--
-- Name: ava_spec_database_tables ava_spec_database_tables_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_database_tables
    ADD CONSTRAINT ava_spec_database_tables_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_database_tables ava_spec_database_tables_spec_id_table_name_schema_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_database_tables
    ADD CONSTRAINT ava_spec_database_tables_spec_id_table_name_schema_name_key UNIQUE (spec_id, table_name, schema_name);


--
-- Name: ava_spec_dependencies ava_spec_dependencies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_dependencies
    ADD CONSTRAINT ava_spec_dependencies_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_dependencies ava_spec_dependencies_source_spec_id_target_spec_id_depende_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_dependencies
    ADD CONSTRAINT ava_spec_dependencies_source_spec_id_target_spec_id_depende_key UNIQUE (source_spec_id, target_spec_id, dependency_type);


--
-- Name: ava_spec_efficiency_ratings ava_spec_efficiency_ratings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_efficiency_ratings
    ADD CONSTRAINT ava_spec_efficiency_ratings_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_efficiency_ratings ava_spec_efficiency_ratings_spec_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_efficiency_ratings
    ADD CONSTRAINT ava_spec_efficiency_ratings_spec_id_key UNIQUE (spec_id);


--
-- Name: ava_spec_enhancements ava_spec_enhancements_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_enhancements
    ADD CONSTRAINT ava_spec_enhancements_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_error_handling ava_spec_error_handling_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_error_handling
    ADD CONSTRAINT ava_spec_error_handling_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_integrations ava_spec_integrations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_integrations
    ADD CONSTRAINT ava_spec_integrations_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_integrations ava_spec_integrations_spec_id_integration_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_integrations
    ADD CONSTRAINT ava_spec_integrations_spec_id_integration_name_key UNIQUE (spec_id, integration_name);


--
-- Name: ava_spec_known_issues ava_spec_known_issues_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_known_issues
    ADD CONSTRAINT ava_spec_known_issues_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_performance_metrics ava_spec_performance_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_performance_metrics
    ADD CONSTRAINT ava_spec_performance_metrics_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_performance_metrics ava_spec_performance_metrics_spec_id_metric_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_performance_metrics
    ADD CONSTRAINT ava_spec_performance_metrics_spec_id_metric_name_key UNIQUE (spec_id, metric_name);


--
-- Name: ava_spec_source_files ava_spec_source_files_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_source_files
    ADD CONSTRAINT ava_spec_source_files_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_source_files ava_spec_source_files_spec_id_file_path_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_source_files
    ADD CONSTRAINT ava_spec_source_files_spec_id_file_path_key UNIQUE (spec_id, file_path);


--
-- Name: ava_spec_tags ava_spec_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_tags
    ADD CONSTRAINT ava_spec_tags_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_tags ava_spec_tags_spec_id_tag_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_tags
    ADD CONSTRAINT ava_spec_tags_spec_id_tag_key UNIQUE (spec_id, tag);


--
-- Name: ava_spec_version_history ava_spec_version_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_version_history
    ADD CONSTRAINT ava_spec_version_history_pkey PRIMARY KEY (id);


--
-- Name: ava_spec_version_history ava_spec_version_history_spec_id_version_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_version_history
    ADD CONSTRAINT ava_spec_version_history_spec_id_version_key UNIQUE (spec_id, version);


--
-- Name: ava_user_goals ava_user_goals_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_user_goals
    ADD CONSTRAINT ava_user_goals_pkey PRIMARY KEY (id);


--
-- Name: ava_user_goals ava_user_goals_user_id_platform_goal_type_goal_name_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_user_goals
    ADD CONSTRAINT ava_user_goals_user_id_platform_goal_type_goal_name_key UNIQUE (user_id, platform, goal_type, goal_name);


--
-- Name: earnings_alerts earnings_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_alerts
    ADD CONSTRAINT earnings_alerts_pkey PRIMARY KEY (id);


--
-- Name: earnings_events earnings_events_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_events
    ADD CONSTRAINT earnings_events_pkey PRIMARY KEY (id);


--
-- Name: earnings_events earnings_events_symbol_earnings_date_fiscal_quarter_fiscal__key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_events
    ADD CONSTRAINT earnings_events_symbol_earnings_date_fiscal_quarter_fiscal__key UNIQUE (symbol, earnings_date, fiscal_quarter, fiscal_year);


--
-- Name: earnings_history earnings_history_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_history
    ADD CONSTRAINT earnings_history_pkey PRIMARY KEY (id);


--
-- Name: earnings_history earnings_history_symbol_report_date_fiscal_quarter_fiscal_y_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_history
    ADD CONSTRAINT earnings_history_symbol_report_date_fiscal_quarter_fiscal_y_key UNIQUE (symbol, report_date, fiscal_quarter, fiscal_year);


--
-- Name: earnings_sync_status earnings_sync_status_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_sync_status
    ADD CONSTRAINT earnings_sync_status_pkey PRIMARY KEY (id);


--
-- Name: earnings_sync_status earnings_sync_status_symbol_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.earnings_sync_status
    ADD CONSTRAINT earnings_sync_status_symbol_key UNIQUE (symbol);


--
-- Name: etfs_universe etfs_universe_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etfs_universe
    ADD CONSTRAINT etfs_universe_pkey PRIMARY KEY (id);


--
-- Name: etfs_universe etfs_universe_symbol_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etfs_universe
    ADD CONSTRAINT etfs_universe_symbol_key UNIQUE (symbol);


--
-- Name: live_odds_snapshots idx_odds_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_odds_snapshots
    ADD CONSTRAINT idx_odds_unique UNIQUE (sport, game_id, snapshot_time);


--
-- Name: kalshi_markets kalshi_markets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_markets
    ADD CONSTRAINT kalshi_markets_pkey PRIMARY KEY (id);


--
-- Name: kalshi_markets kalshi_markets_ticker_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_markets
    ADD CONSTRAINT kalshi_markets_ticker_key UNIQUE (ticker);


--
-- Name: kalshi_predictions kalshi_predictions_market_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_predictions
    ADD CONSTRAINT kalshi_predictions_market_id_key UNIQUE (market_id);


--
-- Name: kalshi_predictions kalshi_predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_predictions
    ADD CONSTRAINT kalshi_predictions_pkey PRIMARY KEY (id);


--
-- Name: kalshi_price_history kalshi_price_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_price_history
    ADD CONSTRAINT kalshi_price_history_pkey PRIMARY KEY (id);


--
-- Name: kalshi_sync_log kalshi_sync_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_sync_log
    ADD CONSTRAINT kalshi_sync_log_pkey PRIMARY KEY (id);


--
-- Name: live_odds_snapshots live_odds_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_odds_snapshots
    ADD CONSTRAINT live_odds_snapshots_pkey PRIMARY KEY (id);


--
-- Name: live_prediction_snapshots live_prediction_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_prediction_snapshots
    ADD CONSTRAINT live_prediction_snapshots_pkey PRIMARY KEY (id);


--
-- Name: model_performance model_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_performance
    ADD CONSTRAINT model_performance_pkey PRIMARY KEY (id);


--
-- Name: nba_games nba_games_game_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nba_games
    ADD CONSTRAINT nba_games_game_id_key UNIQUE (game_id);


--
-- Name: nba_games nba_games_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nba_games
    ADD CONSTRAINT nba_games_pkey PRIMARY KEY (id);


--
-- Name: ncaa_basketball_games ncaa_basketball_games_game_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_basketball_games
    ADD CONSTRAINT ncaa_basketball_games_game_id_key UNIQUE (game_id);


--
-- Name: ncaa_basketball_games ncaa_basketball_games_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_basketball_games
    ADD CONSTRAINT ncaa_basketball_games_pkey PRIMARY KEY (id);


--
-- Name: ncaa_football_games ncaa_football_games_game_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_football_games
    ADD CONSTRAINT ncaa_football_games_game_id_key UNIQUE (game_id);


--
-- Name: ncaa_football_games ncaa_football_games_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ncaa_football_games
    ADD CONSTRAINT ncaa_football_games_pkey PRIMARY KEY (id);


--
-- Name: nfl_alert_history nfl_alert_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history
    ADD CONSTRAINT nfl_alert_history_pkey PRIMARY KEY (id);


--
-- Name: nfl_alert_triggers nfl_alert_triggers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_triggers
    ADD CONSTRAINT nfl_alert_triggers_pkey PRIMARY KEY (id);


--
-- Name: nfl_data_sync_log nfl_data_sync_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_data_sync_log
    ADD CONSTRAINT nfl_data_sync_log_pkey PRIMARY KEY (id);


--
-- Name: nfl_games nfl_games_game_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_games
    ADD CONSTRAINT nfl_games_game_id_key UNIQUE (game_id);


--
-- Name: nfl_games nfl_games_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_games
    ADD CONSTRAINT nfl_games_pkey PRIMARY KEY (id);


--
-- Name: nfl_injuries nfl_injuries_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_injuries
    ADD CONSTRAINT nfl_injuries_pkey PRIMARY KEY (id);


--
-- Name: nfl_kalshi_correlations nfl_kalshi_correlations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_kalshi_correlations
    ADD CONSTRAINT nfl_kalshi_correlations_pkey PRIMARY KEY (id);


--
-- Name: nfl_player_stats nfl_player_stats_game_id_player_id_player_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_player_stats
    ADD CONSTRAINT nfl_player_stats_game_id_player_id_player_name_key UNIQUE (game_id, player_id, player_name);


--
-- Name: nfl_player_stats nfl_player_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_player_stats
    ADD CONSTRAINT nfl_player_stats_pkey PRIMARY KEY (id);


--
-- Name: nfl_plays nfl_plays_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_plays
    ADD CONSTRAINT nfl_plays_pkey PRIMARY KEY (id);


--
-- Name: nfl_plays nfl_plays_play_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_plays
    ADD CONSTRAINT nfl_plays_play_id_key UNIQUE (play_id);


--
-- Name: nfl_social_sentiment nfl_social_sentiment_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_social_sentiment
    ADD CONSTRAINT nfl_social_sentiment_pkey PRIMARY KEY (id);


--
-- Name: odds_history odds_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.odds_history
    ADD CONSTRAINT odds_history_pkey PRIMARY KEY (id);


--
-- Name: portfolio_history portfolio_history_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.portfolio_history
    ADD CONSTRAINT portfolio_history_date_key UNIQUE (date);


--
-- Name: portfolio_history portfolio_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.portfolio_history
    ADD CONSTRAINT portfolio_history_pkey PRIMARY KEY (id);


--
-- Name: prediction_results prediction_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prediction_results
    ADD CONSTRAINT prediction_results_pkey PRIMARY KEY (id);


--
-- Name: prediction_results prediction_results_prediction_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prediction_results
    ADD CONSTRAINT prediction_results_prediction_id_key UNIQUE (prediction_id);


--
-- Name: premium_opportunities premium_opportunities_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_opportunities
    ADD CONSTRAINT premium_opportunities_pkey PRIMARY KEY (id);


--
-- Name: premium_scan_history premium_scan_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_scan_history
    ADD CONSTRAINT premium_scan_history_pkey PRIMARY KEY (id);


--
-- Name: premium_scan_history premium_scan_history_scan_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_scan_history
    ADD CONSTRAINT premium_scan_history_scan_id_key UNIQUE (scan_id);


--
-- Name: scanner_results scanner_results_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.scanner_results
    ADD CONSTRAINT scanner_results_pkey PRIMARY KEY (id);


--
-- Name: scanner_watchlists scanner_watchlists_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_watchlists
    ADD CONSTRAINT scanner_watchlists_pkey PRIMARY KEY (id);


--
-- Name: scanner_watchlists_sync_log scanner_watchlists_sync_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_watchlists_sync_log
    ADD CONSTRAINT scanner_watchlists_sync_log_pkey PRIMARY KEY (id);


--
-- Name: scanner_watchlists scanner_watchlists_watchlist_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_watchlists
    ADD CONSTRAINT scanner_watchlists_watchlist_id_key UNIQUE (watchlist_id);


--
-- Name: stock_ai_scores stock_ai_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_ai_scores
    ADD CONSTRAINT stock_ai_scores_pkey PRIMARY KEY (id);


--
-- Name: stocks_universe stocks_universe_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks_universe
    ADD CONSTRAINT stocks_universe_pkey PRIMARY KEY (id);


--
-- Name: stocks_universe stocks_universe_symbol_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks_universe
    ADD CONSTRAINT stocks_universe_symbol_key UNIQUE (symbol);


--
-- Name: trade_history trade_history_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.trade_history
    ADD CONSTRAINT trade_history_pkey PRIMARY KEY (id);


--
-- Name: trade_journal trade_journal_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.trade_journal
    ADD CONSTRAINT trade_journal_pkey PRIMARY KEY (id);


--
-- Name: tv_symbols_api tv_symbols_api_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_symbols_api
    ADD CONSTRAINT tv_symbols_api_pkey PRIMARY KEY (id);


--
-- Name: tv_symbols_api tv_symbols_api_watchlist_id_symbol_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_symbols_api
    ADD CONSTRAINT tv_symbols_api_watchlist_id_symbol_key UNIQUE (watchlist_id, symbol);


--
-- Name: tv_watchlists_api tv_watchlists_api_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_watchlists_api
    ADD CONSTRAINT tv_watchlists_api_pkey PRIMARY KEY (id);


--
-- Name: tv_watchlists_api tv_watchlists_api_watchlist_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_watchlists_api
    ADD CONSTRAINT tv_watchlists_api_watchlist_id_key UNIQUE (watchlist_id);


--
-- Name: premium_opportunities unique_option; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_opportunities
    ADD CONSTRAINT unique_option UNIQUE (symbol, option_type, strike, expiration);


--
-- Name: user_bets user_bets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_bets
    ADD CONSTRAINT user_bets_pkey PRIMARY KEY (id);


--
-- Name: user_betting_profile user_betting_profile_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_betting_profile
    ADD CONSTRAINT user_betting_profile_pkey PRIMARY KEY (user_id);


--
-- Name: xtrades_notifications xtrades_notifications_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_notifications
    ADD CONSTRAINT xtrades_notifications_pkey PRIMARY KEY (id);


--
-- Name: xtrades_profiles xtrades_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_profiles
    ADD CONSTRAINT xtrades_profiles_pkey PRIMARY KEY (id);


--
-- Name: xtrades_profiles xtrades_profiles_username_key; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_profiles
    ADD CONSTRAINT xtrades_profiles_username_key UNIQUE (username);


--
-- Name: xtrades_sync_log xtrades_sync_log_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_sync_log
    ADD CONSTRAINT xtrades_sync_log_pkey PRIMARY KEY (id);


--
-- Name: xtrades_trades xtrades_trades_pkey; Type: CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_trades
    ADD CONSTRAINT xtrades_trades_pkey PRIMARY KEY (id);


--
-- Name: idx_agent_execution_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_agent_execution_name ON public.agent_execution_log USING btree (agent_name);


--
-- Name: idx_agent_execution_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_agent_execution_time ON public.agent_execution_log USING btree ("timestamp");


--
-- Name: idx_agent_feedback_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_agent_feedback_name ON public.agent_feedback USING btree (agent_name);


--
-- Name: idx_agent_memory_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_agent_memory_name ON public.agent_memory USING btree (agent_name);


--
-- Name: idx_ai_recs_confidence; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ai_recs_confidence ON public.ai_betting_recommendations USING btree (confidence DESC);


--
-- Name: idx_ai_recs_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ai_recs_game ON public.ai_betting_recommendations USING btree (sport, game_id);


--
-- Name: idx_ai_recs_unsettled; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ai_recs_unsettled ON public.ai_betting_recommendations USING btree (is_settled) WHERE (is_settled = false);


--
-- Name: idx_automations_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_automations_category ON public.automations USING btree (category);


--
-- Name: idx_automations_enabled; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_automations_enabled ON public.automations USING btree (is_enabled);


--
-- Name: idx_automations_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_automations_name ON public.automations USING btree (name);


--
-- Name: idx_automations_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_automations_type ON public.automations USING btree (automation_type);


--
-- Name: idx_ava_alert_prefs_category; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alert_prefs_category ON public.ava_alert_preferences USING btree (category);


--
-- Name: idx_ava_alert_prefs_user; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alert_prefs_user ON public.ava_alert_preferences USING btree (user_id, platform);


--
-- Name: idx_ava_alerts_active; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_active ON public.ava_alerts USING btree (is_active, created_at DESC) WHERE (is_active = true);


--
-- Name: idx_ava_alerts_category; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_category ON public.ava_alerts USING btree (category, created_at DESC);


--
-- Name: idx_ava_alerts_fingerprint; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_fingerprint ON public.ava_alerts USING btree (fingerprint);


--
-- Name: idx_ava_alerts_metadata; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_metadata ON public.ava_alerts USING gin (metadata);


--
-- Name: idx_ava_alerts_priority; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_priority ON public.ava_alerts USING btree (priority, is_active);


--
-- Name: idx_ava_alerts_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_alerts_symbol ON public.ava_alerts USING btree (symbol) WHERE (symbol IS NOT NULL);


--
-- Name: idx_ava_deliveries_alert; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_deliveries_alert ON public.ava_alert_deliveries USING btree (alert_id);


--
-- Name: idx_ava_deliveries_channel; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_deliveries_channel ON public.ava_alert_deliveries USING btree (channel, status);


--
-- Name: idx_ava_deliveries_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_deliveries_status ON public.ava_alert_deliveries USING btree (status, created_at DESC);


--
-- Name: idx_ava_goal_history_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goal_history_date ON public.ava_goal_progress_history USING btree (snapshot_date DESC);


--
-- Name: idx_ava_goal_history_goal; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goal_history_goal ON public.ava_goal_progress_history USING btree (goal_id, snapshot_date DESC);


--
-- Name: idx_ava_goals_active; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goals_active ON public.ava_user_goals USING btree (status, user_id) WHERE ((status)::text = 'active'::text);


--
-- Name: idx_ava_goals_period; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goals_period ON public.ava_user_goals USING btree (period_type, start_date);


--
-- Name: idx_ava_goals_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goals_type ON public.ava_user_goals USING btree (goal_type);


--
-- Name: idx_ava_goals_user; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_goals_user ON public.ava_user_goals USING btree (user_id, platform);


--
-- Name: idx_ava_iv_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_iv_date ON public.ava_iv_history USING btree (date DESC);


--
-- Name: idx_ava_iv_rank; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_iv_rank ON public.ava_iv_history USING btree (iv_rank DESC) WHERE (iv_rank IS NOT NULL);


--
-- Name: idx_ava_iv_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_iv_symbol ON public.ava_iv_history USING btree (symbol, date DESC);


--
-- Name: idx_ava_patterns_active; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_patterns_active ON public.ava_learning_patterns USING btree (active, confidence_score DESC) WHERE (active = true);


--
-- Name: idx_ava_patterns_conditions; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_patterns_conditions ON public.ava_learning_patterns USING gin (pattern_conditions);


--
-- Name: idx_ava_patterns_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_patterns_type ON public.ava_learning_patterns USING btree (pattern_type);


--
-- Name: idx_ava_patterns_user; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_patterns_user ON public.ava_learning_patterns USING btree (user_id, platform);


--
-- Name: idx_ava_rate_limits_lookup; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_rate_limits_lookup ON public.ava_alert_rate_limits USING btree (user_id, channel, window_start DESC);


--
-- Name: idx_ava_recs_context; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_context ON public.ava_chat_recommendations USING gin (context_snapshot);


--
-- Name: idx_ava_recs_correct; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_correct ON public.ava_chat_recommendations USING btree (recommendation_correct) WHERE (recommendation_correct IS NOT NULL);


--
-- Name: idx_ava_recs_created; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_created ON public.ava_chat_recommendations USING btree (created_at DESC);


--
-- Name: idx_ava_recs_outcome; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_outcome ON public.ava_chat_recommendations USING btree (actual_outcome) WHERE (actual_outcome IS NOT NULL);


--
-- Name: idx_ava_recs_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_symbol ON public.ava_chat_recommendations USING btree (symbol) WHERE (symbol IS NOT NULL);


--
-- Name: idx_ava_recs_user; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_recs_user ON public.ava_chat_recommendations USING btree (user_id, platform);


--
-- Name: idx_ava_reports_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_reports_date ON public.ava_generated_reports USING btree (report_date DESC);


--
-- Name: idx_ava_reports_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_reports_type ON public.ava_generated_reports USING btree (report_type, report_date DESC);


--
-- Name: idx_ava_scans_started; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_scans_started ON public.ava_opportunity_scans USING btree (started_at DESC);


--
-- Name: idx_ava_scans_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_ava_scans_type ON public.ava_opportunity_scans USING btree (scan_type, started_at DESC);


--
-- Name: idx_earnings_alerts_active; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_alerts_active ON public.earnings_alerts USING btree (is_active, symbol) WHERE (is_active = true);


--
-- Name: idx_earnings_alerts_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_alerts_symbol ON public.earnings_alerts USING btree (symbol);


--
-- Name: idx_earnings_events_confirmed; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_events_confirmed ON public.earnings_events USING btree (is_confirmed, earnings_date) WHERE (is_confirmed = true);


--
-- Name: idx_earnings_events_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_events_date ON public.earnings_events USING btree (earnings_date);


--
-- Name: idx_earnings_events_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_events_symbol ON public.earnings_events USING btree (symbol);


--
-- Name: idx_earnings_events_symbol_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_events_symbol_date ON public.earnings_events USING btree (symbol, earnings_date DESC);


--
-- Name: idx_earnings_history_beat_miss; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_history_beat_miss ON public.earnings_history USING btree (beat_miss) WHERE (beat_miss IS NOT NULL);


--
-- Name: idx_earnings_history_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_history_date ON public.earnings_history USING btree (report_date DESC);


--
-- Name: idx_earnings_history_quarter; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_history_quarter ON public.earnings_history USING btree (fiscal_year, fiscal_quarter);


--
-- Name: idx_earnings_history_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_history_symbol ON public.earnings_history USING btree (symbol);


--
-- Name: idx_earnings_history_symbol_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_earnings_history_symbol_date ON public.earnings_history USING btree (symbol, report_date DESC);


--
-- Name: idx_efficiency_below_threshold; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_efficiency_below_threshold ON public.ava_spec_efficiency_ratings USING btree (overall_rating) WHERE (overall_rating < (7)::numeric);


--
-- Name: idx_efficiency_spec_rating; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_efficiency_spec_rating ON public.ava_spec_efficiency_ratings USING btree (spec_id, overall_rating);


--
-- Name: idx_enhancements_proposed; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_enhancements_proposed ON public.ava_spec_enhancements USING btree (spec_id) WHERE ((status)::text = 'proposed'::text);


--
-- Name: idx_etfs_assets; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_etfs_assets ON public.etfs_universe USING btree (total_assets DESC);


--
-- Name: idx_etfs_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_etfs_category ON public.etfs_universe USING btree (category);


--
-- Name: idx_etfs_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_etfs_symbol ON public.etfs_universe USING btree (symbol);


--
-- Name: idx_executions_automation_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_executions_automation_id ON public.automation_executions USING btree (automation_id);


--
-- Name: idx_executions_automation_status_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_executions_automation_status_time ON public.automation_executions USING btree (automation_id, status, started_at DESC);


--
-- Name: idx_executions_celery_task_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_executions_celery_task_id ON public.automation_executions USING btree (celery_task_id);


--
-- Name: idx_executions_started_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_executions_started_at ON public.automation_executions USING btree (started_at DESC);


--
-- Name: idx_executions_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_executions_status ON public.automation_executions USING btree (status);


--
-- Name: idx_feature_specs_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_category ON public.ava_feature_specs USING btree (category);


--
-- Name: idx_feature_specs_current; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_current ON public.ava_feature_specs USING btree (is_current) WHERE (is_current = true);


--
-- Name: idx_feature_specs_embedding; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_embedding ON public.ava_feature_specs USING gin (embedding);


--
-- Name: idx_feature_specs_feature_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_feature_id ON public.ava_feature_specs USING btree (feature_id);


--
-- Name: idx_feature_specs_fts; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_fts ON public.ava_feature_specs USING gin (to_tsvector('english'::regconfig, (((((COALESCE(feature_name, ''::character varying))::text || ' '::text) || COALESCE(purpose, ''::text)) || ' '::text) || COALESCE(description, ''::text))));


--
-- Name: idx_feature_specs_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_status ON public.ava_feature_specs USING btree (status) WHERE ((status)::text = 'active'::text);


--
-- Name: idx_feature_specs_subcategory; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_subcategory ON public.ava_feature_specs USING btree (category, subcategory);


--
-- Name: idx_feature_specs_technical; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_specs_technical ON public.ava_feature_specs USING gin (technical_details);


--
-- Name: idx_issues_unresolved; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_issues_unresolved ON public.ava_spec_known_issues USING btree (spec_id) WHERE ((status)::text <> 'resolved'::text);


--
-- Name: idx_kalshi_markets_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_active ON public.kalshi_markets USING btree (status, game_date) WHERE ((status)::text = 'open'::text);


--
-- Name: idx_kalshi_markets_away_team; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_away_team ON public.kalshi_markets USING btree (away_team);


--
-- Name: idx_kalshi_markets_close_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_close_time ON public.kalshi_markets USING btree (close_time);


--
-- Name: idx_kalshi_markets_game_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_game_date ON public.kalshi_markets USING btree (game_date);


--
-- Name: idx_kalshi_markets_home_team; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_home_team ON public.kalshi_markets USING btree (home_team);


--
-- Name: idx_kalshi_markets_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_status ON public.kalshi_markets USING btree (status);


--
-- Name: idx_kalshi_markets_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_markets_type ON public.kalshi_markets USING btree (market_type);


--
-- Name: idx_kalshi_predictions_action; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_predictions_action ON public.kalshi_predictions USING btree (recommended_action);


--
-- Name: idx_kalshi_predictions_edge; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_predictions_edge ON public.kalshi_predictions USING btree (edge_percentage DESC);


--
-- Name: idx_kalshi_predictions_market; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_predictions_market ON public.kalshi_predictions USING btree (market_id);


--
-- Name: idx_kalshi_predictions_rank; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_predictions_rank ON public.kalshi_predictions USING btree (overall_rank);


--
-- Name: idx_kalshi_price_history_market; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_price_history_market ON public.kalshi_price_history USING btree (market_id);


--
-- Name: idx_kalshi_price_history_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_price_history_ticker ON public.kalshi_price_history USING btree (ticker);


--
-- Name: idx_kalshi_price_history_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_price_history_time ON public.kalshi_price_history USING btree (snapshot_time);


--
-- Name: idx_kalshi_sync_log_started; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_sync_log_started ON public.kalshi_sync_log USING btree (started_at DESC);


--
-- Name: idx_kalshi_sync_log_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_kalshi_sync_log_type ON public.kalshi_sync_log USING btree (sync_type);


--
-- Name: idx_live_snapshots_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_live_snapshots_game ON public.live_prediction_snapshots USING btree (game_id);


--
-- Name: idx_live_snapshots_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_live_snapshots_time ON public.live_prediction_snapshots USING btree (snapshot_at);


--
-- Name: idx_model_perf_unique; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_model_perf_unique ON public.model_performance USING btree (sport, model_version, period_start, period_end);


--
-- Name: idx_nba_games_live; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_live ON public.nba_games USING btree (is_live) WHERE (is_live = true);


--
-- Name: idx_nba_games_live_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_live_time ON public.nba_games USING btree (is_live, game_time DESC) WHERE (is_live = true);


--
-- Name: idx_nba_games_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_status ON public.nba_games USING btree (game_status);


--
-- Name: idx_nba_games_status_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_status_time ON public.nba_games USING btree (game_status, game_time);


--
-- Name: idx_nba_games_teams; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_teams ON public.nba_games USING btree (home_team, away_team);


--
-- Name: idx_nba_games_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nba_games_time ON public.nba_games USING btree (game_time);


--
-- Name: idx_ncaab_games_live; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaab_games_live ON public.ncaa_basketball_games USING btree (is_live) WHERE (is_live = true);


--
-- Name: idx_ncaab_games_live_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaab_games_live_time ON public.ncaa_basketball_games USING btree (is_live, game_time DESC) WHERE (is_live = true);


--
-- Name: idx_ncaab_games_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaab_games_status ON public.ncaa_basketball_games USING btree (game_status);


--
-- Name: idx_ncaab_games_status_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaab_games_status_time ON public.ncaa_basketball_games USING btree (game_status, game_time);


--
-- Name: idx_ncaab_games_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaab_games_time ON public.ncaa_basketball_games USING btree (game_time);


--
-- Name: idx_ncaaf_games_live; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_live ON public.ncaa_football_games USING btree (is_live) WHERE (is_live = true);


--
-- Name: idx_ncaaf_games_live_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_live_time ON public.ncaa_football_games USING btree (is_live, game_time DESC) WHERE (is_live = true);


--
-- Name: idx_ncaaf_games_ranked; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_ranked ON public.ncaa_football_games USING btree (home_rank, away_rank) WHERE ((home_rank IS NOT NULL) OR (away_rank IS NOT NULL));


--
-- Name: idx_ncaaf_games_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_status ON public.ncaa_football_games USING btree (game_status);


--
-- Name: idx_ncaaf_games_status_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_status_time ON public.ncaa_football_games USING btree (game_status, game_time);


--
-- Name: idx_ncaaf_games_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ncaaf_games_time ON public.ncaa_football_games USING btree (game_time);


--
-- Name: idx_nfl_alert_history_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_history_game ON public.nfl_alert_history USING btree (game_id);


--
-- Name: idx_nfl_alert_history_sent; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_history_sent ON public.nfl_alert_history USING btree (sent_at DESC);


--
-- Name: idx_nfl_alert_history_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_history_status ON public.nfl_alert_history USING btree (delivery_status);


--
-- Name: idx_nfl_alert_history_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_history_type ON public.nfl_alert_history USING btree (alert_type);


--
-- Name: idx_nfl_alert_triggers_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_triggers_active ON public.nfl_alert_triggers USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_nfl_alert_triggers_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_alert_triggers_type ON public.nfl_alert_triggers USING btree (alert_type);


--
-- Name: idx_nfl_games_live; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_live ON public.nfl_games USING btree (is_live) WHERE (is_live = true);


--
-- Name: idx_nfl_games_live_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_live_time ON public.nfl_games USING btree (is_live, game_time DESC) WHERE (is_live = true);


--
-- Name: idx_nfl_games_season_week; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_season_week ON public.nfl_games USING btree (season, week);


--
-- Name: idx_nfl_games_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_status ON public.nfl_games USING btree (game_status);


--
-- Name: idx_nfl_games_status_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_status_time ON public.nfl_games USING btree (game_status, game_time);


--
-- Name: idx_nfl_games_teams; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_teams ON public.nfl_games USING btree (home_team, away_team);


--
-- Name: idx_nfl_games_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_time ON public.nfl_games USING btree (game_time);


--
-- Name: idx_nfl_games_upcoming; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_games_upcoming ON public.nfl_games USING btree (game_time) WHERE ((game_status)::text = ANY ((ARRAY['scheduled'::character varying, 'live'::character varying])::text[]));


--
-- Name: idx_nfl_injuries_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_injuries_active ON public.nfl_injuries USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_nfl_injuries_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_injuries_game ON public.nfl_injuries USING btree (game_id);


--
-- Name: idx_nfl_injuries_player; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_injuries_player ON public.nfl_injuries USING btree (player_id);


--
-- Name: idx_nfl_injuries_team; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_injuries_team ON public.nfl_injuries USING btree (team);


--
-- Name: idx_nfl_kalshi_corr_event; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_kalshi_corr_event ON public.nfl_kalshi_correlations USING btree (event_type);


--
-- Name: idx_nfl_kalshi_corr_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_kalshi_corr_game ON public.nfl_kalshi_correlations USING btree (game_id);


--
-- Name: idx_nfl_kalshi_corr_market; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_kalshi_corr_market ON public.nfl_kalshi_correlations USING btree (kalshi_market_id);


--
-- Name: idx_nfl_kalshi_corr_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_kalshi_corr_timestamp ON public.nfl_kalshi_correlations USING btree (event_timestamp DESC);


--
-- Name: idx_nfl_player_stats_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_player_stats_game ON public.nfl_player_stats USING btree (game_id);


--
-- Name: idx_nfl_player_stats_player; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_player_stats_player ON public.nfl_player_stats USING btree (player_id);


--
-- Name: idx_nfl_player_stats_position; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_player_stats_position ON public.nfl_player_stats USING btree ("position");


--
-- Name: idx_nfl_plays_created; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_plays_created ON public.nfl_plays USING btree (created_at DESC);


--
-- Name: idx_nfl_plays_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_plays_game ON public.nfl_plays USING btree (game_id);


--
-- Name: idx_nfl_plays_scoring; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_plays_scoring ON public.nfl_plays USING btree (is_scoring_play) WHERE (is_scoring_play = true);


--
-- Name: idx_nfl_plays_sequence; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_plays_sequence ON public.nfl_plays USING btree (game_id, sequence_number);


--
-- Name: idx_nfl_plays_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_plays_type ON public.nfl_plays USING btree (play_type);


--
-- Name: idx_nfl_sentiment_entity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sentiment_entity ON public.nfl_social_sentiment USING btree (entity_type, entity_id);


--
-- Name: idx_nfl_sentiment_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sentiment_game ON public.nfl_social_sentiment USING btree (game_id);


--
-- Name: idx_nfl_sentiment_window; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sentiment_window ON public.nfl_social_sentiment USING btree (window_end DESC);


--
-- Name: idx_nfl_sync_log_started; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sync_log_started ON public.nfl_data_sync_log USING btree (started_at DESC);


--
-- Name: idx_nfl_sync_log_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sync_log_status ON public.nfl_data_sync_log USING btree (sync_status);


--
-- Name: idx_nfl_sync_log_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_nfl_sync_log_type ON public.nfl_data_sync_log USING btree (sync_type);


--
-- Name: idx_odds_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_game ON public.live_odds_snapshots USING btree (sport, game_id);


--
-- Name: idx_odds_game_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_game_time ON public.live_odds_snapshots USING btree (sport, game_id, snapshot_time DESC);


--
-- Name: idx_odds_history_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_history_game ON public.odds_history USING btree (game_id);


--
-- Name: idx_odds_history_source; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_history_source ON public.odds_history USING btree (source);


--
-- Name: idx_odds_history_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_history_time ON public.odds_history USING btree (recorded_at);


--
-- Name: idx_odds_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_odds_time ON public.live_odds_snapshots USING btree (snapshot_time);


--
-- Name: idx_portfolio_history_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_portfolio_history_date ON public.portfolio_history USING btree (date);


--
-- Name: idx_prediction_results_correct; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_prediction_results_correct ON public.prediction_results USING btree (was_correct);


--
-- Name: idx_prediction_results_game_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_prediction_results_game_id ON public.prediction_results USING btree (game_id);


--
-- Name: idx_prediction_results_sport; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_prediction_results_sport ON public.prediction_results USING btree (sport);


--
-- Name: idx_prediction_results_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_prediction_results_timestamp ON public.prediction_results USING btree (prediction_timestamp);


--
-- Name: idx_premium_opportunities_annualized; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_annualized ON public.premium_opportunities USING btree (annualized_return DESC);


--
-- Name: idx_premium_opportunities_dte; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_dte ON public.premium_opportunities USING btree (dte);


--
-- Name: idx_premium_opportunities_expiration; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_expiration ON public.premium_opportunities USING btree (expiration);


--
-- Name: idx_premium_opportunities_last_updated; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_last_updated ON public.premium_opportunities USING btree (last_updated DESC);


--
-- Name: idx_premium_opportunities_premium_pct; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_premium_pct ON public.premium_opportunities USING btree (premium_pct DESC);


--
-- Name: idx_premium_opportunities_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_symbol ON public.premium_opportunities USING btree (symbol);


--
-- Name: idx_premium_opportunities_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_opportunities_type ON public.premium_opportunities USING btree (option_type);


--
-- Name: idx_scan_history_created; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scan_history_created ON public.premium_scan_history USING btree (created_at DESC);


--
-- Name: idx_scan_history_scan_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scan_history_scan_id ON public.premium_scan_history USING btree (scan_id);


--
-- Name: idx_scanner_results_annual_return; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_scanner_results_annual_return ON public.scanner_results USING btree (annual_return DESC);


--
-- Name: idx_scanner_results_scanned_at; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_scanner_results_scanned_at ON public.scanner_results USING btree (scanned_at DESC);


--
-- Name: idx_scanner_results_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_scanner_results_symbol ON public.scanner_results USING btree (symbol);


--
-- Name: idx_scanner_watchlists_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scanner_watchlists_active ON public.scanner_watchlists USING btree (is_active);


--
-- Name: idx_scanner_watchlists_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scanner_watchlists_category ON public.scanner_watchlists USING btree (category);


--
-- Name: idx_scanner_watchlists_sort; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scanner_watchlists_sort ON public.scanner_watchlists USING btree (sort_order, name);


--
-- Name: idx_scanner_watchlists_source; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_scanner_watchlists_source ON public.scanner_watchlists USING btree (source);


--
-- Name: idx_spec_db_tables_owner; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_db_tables_owner ON public.ava_spec_database_tables USING btree (is_owner) WHERE (is_owner = true);


--
-- Name: idx_spec_db_tables_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_db_tables_spec ON public.ava_spec_database_tables USING btree (spec_id);


--
-- Name: idx_spec_db_tables_table; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_db_tables_table ON public.ava_spec_database_tables USING btree (table_name);


--
-- Name: idx_spec_deps_critical; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_deps_critical ON public.ava_spec_dependencies USING btree (is_critical) WHERE (is_critical = true);


--
-- Name: idx_spec_deps_source; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_deps_source ON public.ava_spec_dependencies USING btree (source_spec_id);


--
-- Name: idx_spec_deps_target; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_deps_target ON public.ava_spec_dependencies USING btree (target_spec_id);


--
-- Name: idx_spec_deps_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_deps_type ON public.ava_spec_dependencies USING btree (dependency_type);


--
-- Name: idx_spec_efficiency_low; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_efficiency_low ON public.ava_spec_efficiency_ratings USING btree (overall_rating) WHERE (overall_rating < 7.0);


--
-- Name: idx_spec_efficiency_metrics; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_efficiency_metrics ON public.ava_spec_efficiency_ratings USING gin (metrics);


--
-- Name: idx_spec_efficiency_overall; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_efficiency_overall ON public.ava_spec_efficiency_ratings USING btree (overall_rating DESC);


--
-- Name: idx_spec_efficiency_priority; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_efficiency_priority ON public.ava_spec_efficiency_ratings USING btree (priority_level);


--
-- Name: idx_spec_efficiency_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_efficiency_spec ON public.ava_spec_efficiency_ratings USING btree (spec_id);


--
-- Name: idx_spec_endpoints_method; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_endpoints_method ON public.ava_spec_api_endpoints USING btree (method, path);


--
-- Name: idx_spec_endpoints_path; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_endpoints_path ON public.ava_spec_api_endpoints USING btree (path);


--
-- Name: idx_spec_endpoints_router; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_endpoints_router ON public.ava_spec_api_endpoints USING btree (router_name);


--
-- Name: idx_spec_endpoints_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_endpoints_spec ON public.ava_spec_api_endpoints USING btree (spec_id);


--
-- Name: idx_spec_enhancements_open; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_enhancements_open ON public.ava_spec_enhancements USING btree (priority, status) WHERE ((status)::text = ANY ((ARRAY['proposed'::character varying, 'approved'::character varying])::text[]));


--
-- Name: idx_spec_enhancements_priority; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_enhancements_priority ON public.ava_spec_enhancements USING btree (priority);


--
-- Name: idx_spec_enhancements_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_enhancements_spec ON public.ava_spec_enhancements USING btree (spec_id);


--
-- Name: idx_spec_enhancements_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_enhancements_status ON public.ava_spec_enhancements USING btree (status);


--
-- Name: idx_spec_errors_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_errors_spec ON public.ava_spec_error_handling USING btree (spec_id);


--
-- Name: idx_spec_errors_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_errors_type ON public.ava_spec_error_handling USING btree (error_type);


--
-- Name: idx_spec_history_created; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_history_created ON public.ava_spec_version_history USING btree (created_at DESC);


--
-- Name: idx_spec_history_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_history_spec ON public.ava_spec_version_history USING btree (spec_id);


--
-- Name: idx_spec_history_version; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_history_version ON public.ava_spec_version_history USING btree (version);


--
-- Name: idx_spec_integrations_critical; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_integrations_critical ON public.ava_spec_integrations USING btree (is_critical) WHERE (is_critical = true);


--
-- Name: idx_spec_integrations_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_integrations_name ON public.ava_spec_integrations USING btree (integration_name);


--
-- Name: idx_spec_integrations_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_integrations_spec ON public.ava_spec_integrations USING btree (spec_id);


--
-- Name: idx_spec_integrations_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_integrations_type ON public.ava_spec_integrations USING btree (integration_type);


--
-- Name: idx_spec_issues_severity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_issues_severity ON public.ava_spec_known_issues USING btree (severity);


--
-- Name: idx_spec_issues_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_issues_spec ON public.ava_spec_known_issues USING btree (spec_id);


--
-- Name: idx_spec_issues_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_issues_status ON public.ava_spec_known_issues USING btree (status) WHERE ((status)::text <> 'resolved'::text);


--
-- Name: idx_spec_issues_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_issues_type ON public.ava_spec_known_issues USING btree (issue_type);


--
-- Name: idx_spec_perf_misses; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_perf_misses ON public.ava_spec_performance_metrics USING btree (meets_target) WHERE (meets_target = false);


--
-- Name: idx_spec_perf_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_perf_spec ON public.ava_spec_performance_metrics USING btree (spec_id);


--
-- Name: idx_spec_perf_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_perf_type ON public.ava_spec_performance_metrics USING btree (metric_type);


--
-- Name: idx_spec_source_files_path; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_source_files_path ON public.ava_spec_source_files USING btree (file_path);


--
-- Name: idx_spec_source_files_primary; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_source_files_primary ON public.ava_spec_source_files USING btree (spec_id, is_primary) WHERE (is_primary = true);


--
-- Name: idx_spec_source_files_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_source_files_spec ON public.ava_spec_source_files USING btree (spec_id);


--
-- Name: idx_spec_source_files_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_source_files_type ON public.ava_spec_source_files USING btree (file_type);


--
-- Name: idx_spec_tags_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_tags_category ON public.ava_spec_tags USING btree (tag_category, tag);


--
-- Name: idx_spec_tags_spec; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_tags_spec ON public.ava_spec_tags USING btree (spec_id);


--
-- Name: idx_spec_tags_tag; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spec_tags_tag ON public.ava_spec_tags USING btree (tag);


--
-- Name: idx_specs_current_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_specs_current_status ON public.ava_feature_specs USING btree (is_current, status) WHERE ((is_current = true) AND ((status)::text = 'active'::text));


--
-- Name: idx_specs_list_covering; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_specs_list_covering ON public.ava_feature_specs USING btree (id, feature_name, category, purpose) WHERE (is_current = true);


--
-- Name: idx_state_log_automation; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_state_log_automation ON public.automation_state_log USING btree (automation_id, changed_at DESC);


--
-- Name: idx_stock_scores_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stock_scores_symbol ON public.stock_ai_scores USING btree (symbol);


--
-- Name: idx_stock_scores_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stock_scores_time ON public.stock_ai_scores USING btree (scored_at DESC);


--
-- Name: idx_stocks_market_cap; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stocks_market_cap ON public.stocks_universe USING btree (market_cap DESC);


--
-- Name: idx_stocks_sector; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stocks_sector ON public.stocks_universe USING btree (sector);


--
-- Name: idx_stocks_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stocks_symbol ON public.stocks_universe USING btree (symbol);


--
-- Name: idx_stocks_volume; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_stocks_volume ON public.stocks_universe USING btree (avg_volume_10d DESC);


--
-- Name: idx_sync_status_failed; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_sync_status_failed ON public.earnings_sync_status USING btree (last_sync_status) WHERE ((last_sync_status)::text = 'failed'::text);


--
-- Name: idx_sync_status_next_sync; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_sync_status_next_sync ON public.earnings_sync_status USING btree (next_sync_at) WHERE (next_sync_at IS NOT NULL);


--
-- Name: idx_sync_status_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_sync_status_symbol ON public.earnings_sync_status USING btree (symbol);


--
-- Name: idx_trade_history_close_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_history_close_date ON public.trade_history USING btree (close_date DESC NULLS LAST);


--
-- Name: idx_trade_history_open_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_history_open_date ON public.trade_history USING btree (open_date DESC);


--
-- Name: idx_trade_history_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_history_status ON public.trade_history USING btree (status);


--
-- Name: idx_trade_history_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_history_symbol ON public.trade_history USING btree (symbol);


--
-- Name: idx_trade_journal_closed_at; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_journal_closed_at ON public.trade_journal USING btree (closed_at DESC NULLS LAST);


--
-- Name: idx_trade_journal_opened_at; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_journal_opened_at ON public.trade_journal USING btree (opened_at DESC);


--
-- Name: idx_trade_journal_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_journal_status ON public.trade_journal USING btree (status);


--
-- Name: idx_trade_journal_strategy; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_journal_strategy ON public.trade_journal USING btree (strategy);


--
-- Name: idx_trade_journal_symbol; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_trade_journal_symbol ON public.trade_journal USING btree (symbol);


--
-- Name: idx_tv_symbols_api_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tv_symbols_api_symbol ON public.tv_symbols_api USING btree (symbol);


--
-- Name: idx_tv_symbols_api_watchlist; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tv_symbols_api_watchlist ON public.tv_symbols_api USING btree (watchlist_id);


--
-- Name: idx_user_bets_game; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_user_bets_game ON public.user_bets USING btree (game_id);


--
-- Name: idx_user_bets_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_user_bets_user ON public.user_bets USING btree (user_id);


--
-- Name: idx_xtrades_notifications_sent_at; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_notifications_sent_at ON public.xtrades_notifications USING btree (sent_at DESC);


--
-- Name: idx_xtrades_notifications_trade_id; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_notifications_trade_id ON public.xtrades_notifications USING btree (trade_id);


--
-- Name: idx_xtrades_notifications_trade_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_notifications_trade_type ON public.xtrades_notifications USING btree (trade_id, notification_type);


--
-- Name: idx_xtrades_notifications_type; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_notifications_type ON public.xtrades_notifications USING btree (notification_type);


--
-- Name: idx_xtrades_profiles_active; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_profiles_active ON public.xtrades_profiles USING btree (active);


--
-- Name: idx_xtrades_profiles_last_sync; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_profiles_last_sync ON public.xtrades_profiles USING btree (last_sync DESC);


--
-- Name: idx_xtrades_profiles_username; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_profiles_username ON public.xtrades_profiles USING btree (username);


--
-- Name: idx_xtrades_sync_log_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_sync_log_status ON public.xtrades_sync_log USING btree (status);


--
-- Name: idx_xtrades_sync_log_timestamp; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_sync_log_timestamp ON public.xtrades_sync_log USING btree (sync_timestamp DESC);


--
-- Name: idx_xtrades_trades_alert_id; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_alert_id ON public.xtrades_trades USING btree (xtrades_alert_id);


--
-- Name: idx_xtrades_trades_alert_timestamp; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_alert_timestamp ON public.xtrades_trades USING btree (alert_timestamp DESC);


--
-- Name: idx_xtrades_trades_entry_date; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_entry_date ON public.xtrades_trades USING btree (entry_date DESC);


--
-- Name: idx_xtrades_trades_profile_id; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_profile_id ON public.xtrades_trades USING btree (profile_id);


--
-- Name: idx_xtrades_trades_profile_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_profile_status ON public.xtrades_trades USING btree (profile_id, status);


--
-- Name: idx_xtrades_trades_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_status ON public.xtrades_trades USING btree (status);


--
-- Name: idx_xtrades_trades_strategy; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_strategy ON public.xtrades_trades USING btree (strategy);


--
-- Name: idx_xtrades_trades_ticker; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_ticker ON public.xtrades_trades USING btree (ticker);


--
-- Name: idx_xtrades_trades_ticker_status; Type: INDEX; Schema: public; Owner: adam
--

CREATE INDEX idx_xtrades_trades_ticker_status ON public.xtrades_trades USING btree (ticker, status);


--
-- Name: ava_alert_preferences ava_alert_prefs_updated_at; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER ava_alert_prefs_updated_at BEFORE UPDATE ON public.ava_alert_preferences FOR EACH ROW EXECUTE FUNCTION public.update_ava_advisor_updated_at();


--
-- Name: ava_feature_specs ava_feature_specs_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_feature_specs_updated_at BEFORE UPDATE ON public.ava_feature_specs FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_user_goals ava_goals_updated_at; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER ava_goals_updated_at BEFORE UPDATE ON public.ava_user_goals FOR EACH ROW EXECUTE FUNCTION public.update_ava_advisor_updated_at();


--
-- Name: ava_learning_patterns ava_patterns_updated_at; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER ava_patterns_updated_at BEFORE UPDATE ON public.ava_learning_patterns FOR EACH ROW EXECUTE FUNCTION public.update_ava_advisor_updated_at();


--
-- Name: ava_spec_dependencies ava_spec_dependencies_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_dependencies_updated_at BEFORE UPDATE ON public.ava_spec_dependencies FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_efficiency_ratings ava_spec_efficiency_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_efficiency_updated_at BEFORE UPDATE ON public.ava_spec_efficiency_ratings FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_enhancements ava_spec_enhancements_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_enhancements_updated_at BEFORE UPDATE ON public.ava_spec_enhancements FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_integrations ava_spec_integrations_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_integrations_updated_at BEFORE UPDATE ON public.ava_spec_integrations FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_known_issues ava_spec_issues_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_issues_updated_at BEFORE UPDATE ON public.ava_spec_known_issues FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_performance_metrics ava_spec_perf_metrics_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_perf_metrics_updated_at BEFORE UPDATE ON public.ava_spec_performance_metrics FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: ava_spec_source_files ava_spec_source_files_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ava_spec_source_files_updated_at BEFORE UPDATE ON public.ava_spec_source_files FOR EACH ROW EXECUTE FUNCTION public.update_spec_updated_at();


--
-- Name: scanner_watchlists scanner_watchlists_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER scanner_watchlists_updated_at BEFORE UPDATE ON public.scanner_watchlists FOR EACH ROW EXECUTE FUNCTION public.update_scanner_watchlists_timestamp();


--
-- Name: earnings_events update_earnings_events_timestamp; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER update_earnings_events_timestamp BEFORE UPDATE ON public.earnings_events FOR EACH ROW EXECUTE FUNCTION public.update_modified_column();


--
-- Name: earnings_history update_earnings_history_timestamp; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER update_earnings_history_timestamp BEFORE UPDATE ON public.earnings_history FOR EACH ROW EXECUTE FUNCTION public.update_modified_column();


--
-- Name: earnings_sync_status update_earnings_sync_status_timestamp; Type: TRIGGER; Schema: public; Owner: adam
--

CREATE TRIGGER update_earnings_sync_status_timestamp BEFORE UPDATE ON public.earnings_sync_status FOR EACH ROW EXECUTE FUNCTION public.update_modified_column();


--
-- Name: nba_games update_nba_games_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_nba_games_timestamp BEFORE UPDATE ON public.nba_games FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_column();


--
-- Name: ncaa_basketball_games update_ncaab_games_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_ncaab_games_timestamp BEFORE UPDATE ON public.ncaa_basketball_games FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_column();


--
-- Name: ncaa_football_games update_ncaaf_games_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_ncaaf_games_timestamp BEFORE UPDATE ON public.ncaa_football_games FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_column();


--
-- Name: nfl_games update_nfl_games_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_nfl_games_timestamp BEFORE UPDATE ON public.nfl_games FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_timestamp();


--
-- Name: nfl_injuries update_nfl_injuries_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_nfl_injuries_timestamp BEFORE UPDATE ON public.nfl_injuries FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_timestamp();


--
-- Name: nfl_player_stats update_nfl_player_stats_timestamp; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_nfl_player_stats_timestamp BEFORE UPDATE ON public.nfl_player_stats FOR EACH ROW EXECUTE FUNCTION public.update_last_updated_timestamp();


--
-- Name: automation_executions automation_executions_automation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_executions
    ADD CONSTRAINT automation_executions_automation_id_fkey FOREIGN KEY (automation_id) REFERENCES public.automations(id) ON DELETE CASCADE;


--
-- Name: automation_state_log automation_state_log_automation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.automation_state_log
    ADD CONSTRAINT automation_state_log_automation_id_fkey FOREIGN KEY (automation_id) REFERENCES public.automations(id) ON DELETE CASCADE;


--
-- Name: ava_alert_deliveries ava_alert_deliveries_alert_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_alert_deliveries
    ADD CONSTRAINT ava_alert_deliveries_alert_id_fkey FOREIGN KEY (alert_id) REFERENCES public.ava_alerts(id) ON DELETE CASCADE;


--
-- Name: ava_goal_progress_history ava_goal_progress_history_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.ava_goal_progress_history
    ADD CONSTRAINT ava_goal_progress_history_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.ava_user_goals(id) ON DELETE CASCADE;


--
-- Name: ava_spec_api_endpoints ava_spec_api_endpoints_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_api_endpoints
    ADD CONSTRAINT ava_spec_api_endpoints_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_database_tables ava_spec_database_tables_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_database_tables
    ADD CONSTRAINT ava_spec_database_tables_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_dependencies ava_spec_dependencies_source_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_dependencies
    ADD CONSTRAINT ava_spec_dependencies_source_spec_id_fkey FOREIGN KEY (source_spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_dependencies ava_spec_dependencies_target_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_dependencies
    ADD CONSTRAINT ava_spec_dependencies_target_spec_id_fkey FOREIGN KEY (target_spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_efficiency_ratings ava_spec_efficiency_ratings_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_efficiency_ratings
    ADD CONSTRAINT ava_spec_efficiency_ratings_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_enhancements ava_spec_enhancements_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_enhancements
    ADD CONSTRAINT ava_spec_enhancements_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_error_handling ava_spec_error_handling_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_error_handling
    ADD CONSTRAINT ava_spec_error_handling_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_integrations ava_spec_integrations_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_integrations
    ADD CONSTRAINT ava_spec_integrations_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_known_issues ava_spec_known_issues_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_known_issues
    ADD CONSTRAINT ava_spec_known_issues_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_performance_metrics ava_spec_performance_metrics_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_performance_metrics
    ADD CONSTRAINT ava_spec_performance_metrics_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_source_files ava_spec_source_files_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_source_files
    ADD CONSTRAINT ava_spec_source_files_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_tags ava_spec_tags_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_tags
    ADD CONSTRAINT ava_spec_tags_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: ava_spec_version_history ava_spec_version_history_spec_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ava_spec_version_history
    ADD CONSTRAINT ava_spec_version_history_spec_id_fkey FOREIGN KEY (spec_id) REFERENCES public.ava_feature_specs(id) ON DELETE CASCADE;


--
-- Name: kalshi_predictions kalshi_predictions_market_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_predictions
    ADD CONSTRAINT kalshi_predictions_market_id_fkey FOREIGN KEY (market_id) REFERENCES public.kalshi_markets(id) ON DELETE CASCADE;


--
-- Name: kalshi_price_history kalshi_price_history_market_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kalshi_price_history
    ADD CONSTRAINT kalshi_price_history_market_id_fkey FOREIGN KEY (market_id) REFERENCES public.kalshi_markets(id) ON DELETE CASCADE;


--
-- Name: nfl_alert_history nfl_alert_history_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history
    ADD CONSTRAINT nfl_alert_history_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE SET NULL;


--
-- Name: nfl_alert_history nfl_alert_history_kalshi_market_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history
    ADD CONSTRAINT nfl_alert_history_kalshi_market_id_fkey FOREIGN KEY (kalshi_market_id) REFERENCES public.kalshi_markets(id) ON DELETE SET NULL;


--
-- Name: nfl_alert_history nfl_alert_history_play_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history
    ADD CONSTRAINT nfl_alert_history_play_id_fkey FOREIGN KEY (play_id) REFERENCES public.nfl_plays(id) ON DELETE SET NULL;


--
-- Name: nfl_alert_history nfl_alert_history_trigger_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_alert_history
    ADD CONSTRAINT nfl_alert_history_trigger_id_fkey FOREIGN KEY (trigger_id) REFERENCES public.nfl_alert_triggers(id) ON DELETE SET NULL;


--
-- Name: nfl_injuries nfl_injuries_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_injuries
    ADD CONSTRAINT nfl_injuries_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE SET NULL;


--
-- Name: nfl_kalshi_correlations nfl_kalshi_correlations_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_kalshi_correlations
    ADD CONSTRAINT nfl_kalshi_correlations_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE CASCADE;


--
-- Name: nfl_kalshi_correlations nfl_kalshi_correlations_kalshi_market_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_kalshi_correlations
    ADD CONSTRAINT nfl_kalshi_correlations_kalshi_market_id_fkey FOREIGN KEY (kalshi_market_id) REFERENCES public.kalshi_markets(id) ON DELETE CASCADE;


--
-- Name: nfl_kalshi_correlations nfl_kalshi_correlations_play_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_kalshi_correlations
    ADD CONSTRAINT nfl_kalshi_correlations_play_id_fkey FOREIGN KEY (play_id) REFERENCES public.nfl_plays(id) ON DELETE SET NULL;


--
-- Name: nfl_player_stats nfl_player_stats_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_player_stats
    ADD CONSTRAINT nfl_player_stats_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE CASCADE;


--
-- Name: nfl_plays nfl_plays_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_plays
    ADD CONSTRAINT nfl_plays_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE CASCADE;


--
-- Name: nfl_social_sentiment nfl_social_sentiment_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nfl_social_sentiment
    ADD CONSTRAINT nfl_social_sentiment_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nfl_games(id) ON DELETE CASCADE;


--
-- Name: tv_symbols_api tv_symbols_api_watchlist_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tv_symbols_api
    ADD CONSTRAINT tv_symbols_api_watchlist_id_fkey FOREIGN KEY (watchlist_id) REFERENCES public.tv_watchlists_api(watchlist_id) ON DELETE CASCADE;


--
-- Name: user_bets user_bets_prediction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_bets
    ADD CONSTRAINT user_bets_prediction_id_fkey FOREIGN KEY (prediction_id) REFERENCES public.prediction_results(prediction_id);


--
-- Name: user_bets user_bets_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_bets
    ADD CONSTRAINT user_bets_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.user_betting_profile(user_id);


--
-- Name: xtrades_notifications xtrades_notifications_trade_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_notifications
    ADD CONSTRAINT xtrades_notifications_trade_id_fkey FOREIGN KEY (trade_id) REFERENCES public.xtrades_trades(id) ON DELETE CASCADE;


--
-- Name: xtrades_trades xtrades_trades_profile_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: adam
--

ALTER TABLE ONLY public.xtrades_trades
    ADD CONSTRAINT xtrades_trades_profile_id_fkey FOREIGN KEY (profile_id) REFERENCES public.xtrades_profiles(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

