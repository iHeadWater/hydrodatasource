"""
Tests for FloodEventDatasource class and related functions.

Author: Wenyu Ouyang  
Date: 2025-08-07
"""

import numpy as np
from typing import Dict, List
from hydrodatasource.reader.floodevent import FloodEventDatasource, check_event_data_nan


class TestFloodEventDatasourceCheckEventDataNan:
    """Test cases for FloodEventDatasource.check_event_data_nan method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal datasource instance just for testing the method
        self.datasource = type('FloodEventDatasource', (), {
            'warmup_length': 5,
            'net_rain_key': 'P_eff',
            'obs_flow_key': 'Q_obs_eff',
            'check_event_data_nan': FloodEventDatasource.check_event_data_nan
        })()
        
    def create_test_event(
        self, 
        p_eff: np.ndarray, 
        q_obs: np.ndarray,
        filepath: str = "test_event.csv",
        flood_event_markers: np.ndarray = None
    ) -> Dict:
        """Create a test event dictionary."""
        event = {
            "filepath": filepath,
            "P_eff": p_eff,
            "Q_obs_eff": q_obs,
        }
        if flood_event_markers is not None:
            event["flood_event_markers"] = flood_event_markers
        return event
    
    def test_no_nan_values_exclude_warmup_false(self):
        """Test with no NaN values and exclude_warmup=False."""
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        event = self.create_test_event(p_eff, q_obs)
        
        # Should not raise any exception
        self.datasource.check_event_data_nan([event], exclude_warmup=False)
    
    def test_no_nan_values_exclude_warmup_true_with_markers(self):
        """Test with no NaN values, exclude_warmup=True, and flood_event_markers."""
        # Setup data with warmup period
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # First 5 are warmup (0), last 3 are actual event (>0)
        markers = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        
        event = self.create_test_event(p_eff, q_obs, flood_event_markers=markers)
        
        # Should not raise any exception
        self.datasource.check_event_data_nan([event], exclude_warmup=True)
    
    def test_no_nan_values_exclude_warmup_true_without_markers(self):
        """Test with no NaN values, exclude_warmup=True, and no flood_event_markers."""
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        event = self.create_test_event(p_eff, q_obs)
        
        # Should not raise any exception (excludes first 5 warmup points)
        self.datasource.check_event_data_nan([event], exclude_warmup=True)
    
    def test_nan_in_warmup_period_excluded(self):
        """Test with NaN in warmup period but exclude_warmup=True."""
        # NaN values in warmup period (first 5 elements)
        p_eff = np.array([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        markers = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # First 5 are warmup
        
        event = self.create_test_event(p_eff, q_obs, flood_event_markers=markers)
        
        # Should not raise exception because NaN is in warmup period
        self.datasource.check_event_data_nan([event], exclude_warmup=True)
    
    def test_nan_in_warmup_period_not_excluded(self):
        """Test with NaN in warmup period but exclude_warmup=False."""
        p_eff = np.array([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        event = self.create_test_event(p_eff, q_obs)
        
        # Should raise exception because NaN is checked
        try:
            self.datasource.check_event_data_nan([event], exclude_warmup=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in P_eff" in str(e)
    
    def test_nan_in_event_period_with_markers(self):
        """Test with NaN in actual event period (should raise exception)."""
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        markers = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # NaN is in event period
        
        event = self.create_test_event(p_eff, q_obs, flood_event_markers=markers)
        
        # Should raise exception because NaN is in event period
        try:
            self.datasource.check_event_data_nan([event], exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in P_eff" in str(e)
    
    def test_nan_in_q_obs_event_period(self):
        """Test with NaN in Q_obs in actual event period."""
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, 0.8])
        markers = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # NaN is in event period
        
        event = self.create_test_event(p_eff, q_obs, flood_event_markers=markers)
        
        # Should raise exception because NaN is in Q_obs in event period
        try:
            self.datasource.check_event_data_nan([event], exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in Q_obs_eff" in str(e)
    
    def test_nan_in_event_period_without_markers(self):
        """Test with NaN in event period using warmup_length fallback."""
        # NaN in position 6 (should be in event period when warmup_length=5)
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        event = self.create_test_event(p_eff, q_obs)
        
        # Should raise exception because NaN is after warmup period
        try:
            self.datasource.check_event_data_nan([event], exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in P_eff" in str(e)
    
    def test_multiple_events_mixed_conditions(self):
        """Test with multiple events having different conditions."""
        # Event 1: No NaN
        event1 = self.create_test_event(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            filepath="event1.csv"
        )
        
        # Event 2: NaN in warmup (should be OK when exclude_warmup=True)
        markers2 = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        event2 = self.create_test_event(
            np.array([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            filepath="event2.csv",
            flood_event_markers=markers2
        )
        
        # Should not raise exception
        self.datasource.check_event_data_nan([event1, event2], exclude_warmup=True)
    
    def test_multiple_events_with_nan_in_event_period(self):
        """Test with multiple events where one has NaN in event period."""
        # Event 1: No NaN
        event1 = self.create_test_event(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            filepath="event1.csv"
        )
        
        # Event 2: NaN in event period (should raise exception)
        markers2 = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        event2 = self.create_test_event(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            filepath="event2.csv",
            flood_event_markers=markers2
        )
        
        # Should raise exception for event2
        try:
            self.datasource.check_event_data_nan([event1, event2], exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "event2.csv" in str(e) and "has NaN in P_eff" in str(e)
    
    def test_missing_data_arrays(self):
        """Test with events that have missing P_eff or Q_obs arrays."""
        # Event with missing P_eff
        event1 = {"filepath": "event1.csv", "Q_obs_eff": np.array([1, 2, 3])}
        
        # Event with missing Q_obs_eff
        event2 = {"filepath": "event2.csv", "P_eff": np.array([1, 2, 3])}
        
        # Event with both arrays
        event3 = self.create_test_event(
            np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), "event3.csv"
        )
        
        # Should not raise exception (missing arrays are skipped)
        self.datasource.check_event_data_nan([event1, event2, event3], exclude_warmup=False)
    
    def test_custom_key_names(self):
        """Test with custom key names for rainfall and flow data."""
        # Create datasource with custom key names
        custom_datasource = type('FloodEventDatasource', (), {
            'warmup_length': 3,
            'net_rain_key': 'custom_rain',
            'obs_flow_key': 'custom_flow',
            'check_event_data_nan': FloodEventDatasource.check_event_data_nan
        })()
        
        # Create event with custom key names
        event = {
            "filepath": "custom_event.csv",
            "custom_rain": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            "custom_flow": np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        }
        
        # Should raise exception due to NaN in custom_rain
        try:
            custom_datasource.check_event_data_nan([event], exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in custom_rain" in str(e)


class TestCheckEventDataNanStandaloneFunction:
    """Test cases for the standalone check_event_data_nan function."""
    
    def test_backward_compatibility_basic(self):
        """Test basic backward compatibility with original function interface."""
        p_eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event = {"filepath": "test.csv", "P_eff": p_eff, "Q_obs_eff": q_obs}
        
        # Should not raise exception (original behavior)
        check_event_data_nan([event])
    
    def test_backward_compatibility_with_nan(self):
        """Test backward compatibility when NaN is present."""
        p_eff = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event = {"filepath": "test.csv", "P_eff": p_eff, "Q_obs_eff": q_obs}
        
        # Should raise exception (original behavior)
        try:
            check_event_data_nan([event])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in P_eff" in str(e)
    
    def test_new_exclude_warmup_functionality(self):
        """Test new exclude_warmup functionality in standalone function."""
        # Create data with NaN in warmup period
        p_eff = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        markers = np.array([0, 0, 1, 1, 1])  # First 2 are warmup
        event = {
            "filepath": "test.csv", 
            "P_eff": p_eff, 
            "Q_obs_eff": q_obs,
            "flood_event_markers": markers
        }
        
        # Should not raise exception because NaN is in warmup period
        check_event_data_nan([event], exclude_warmup=True)
    
    def test_new_exclude_warmup_with_warmup_length(self):
        """Test exclude_warmup functionality using warmup_length parameter."""
        # Create data with NaN in warmup period
        p_eff = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        q_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event = {"filepath": "test.csv", "P_eff": p_eff, "Q_obs_eff": q_obs}
        
        # Should not raise exception because NaN is in first 2 warmup points
        check_event_data_nan([event], warmup_length=2, exclude_warmup=True)
        
        # Should raise exception if warmup_length is too small
        try:
            check_event_data_nan([event], warmup_length=1, exclude_warmup=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in P_eff" in str(e)
    
    def test_custom_key_names_in_standalone(self):
        """Test custom key names in standalone function."""
        rain_data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        flow_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event = {
            "filepath": "test.csv",
            "custom_rain": rain_data,
            "custom_flow": flow_data
        }
        
        # Should raise exception due to NaN in custom_rain
        try:
            check_event_data_nan(
                [event], 
                net_rain_key="custom_rain", 
                obs_flow_key="custom_flow"
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "has NaN in custom_rain" in str(e)


if __name__ == "__main__":
    # Run basic tests
    test_class = TestFloodEventDatasourceCheckEventDataNan()
    test_class.setup_method()
    
    print("Running basic tests...")
    test_class.test_no_nan_values_exclude_warmup_false()
    print("OK test_no_nan_values_exclude_warmup_false passed")
    
    test_class.test_no_nan_values_exclude_warmup_true_with_markers()
    print("OK test_no_nan_values_exclude_warmup_true_with_markers passed")
    
    test_class.test_nan_in_warmup_period_excluded()
    print("OK test_nan_in_warmup_period_excluded passed")
    
    print("All basic tests passed!")
    
    # Test standalone function
    standalone_test = TestCheckEventDataNanStandaloneFunction()
    standalone_test.test_backward_compatibility_basic()
    print("OK test_backward_compatibility_basic passed")
    
    standalone_test.test_new_exclude_warmup_functionality()
    print("OK test_new_exclude_warmup_functionality passed")
    
    print("All tests completed successfully!")