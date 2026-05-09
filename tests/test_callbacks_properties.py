"""
Property-based tests for the callbacks module.

**Validates: Requirements 11.1, 11.4**
"""

from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st

from crew.callbacks import ActivityEvent, EventType


class TestActivityEventStructuralCompleteness:
    """Property 9: ActivityEvent structural completeness.

    For any combination of valid EventType, non-empty agent_name string,
    non-empty ticker string, message string, and datetime timestamp,
    constructing an ActivityEvent SHALL produce an object where all five
    fields are accessible and match the input values exactly.

    **Validates: Requirements 11.1, 11.4**
    """

    @given(
        event_type=st.sampled_from(EventType),
        agent_name=st.text(min_size=1),
        ticker=st.text(min_size=1),
        message=st.text(),
        timestamp=st.datetimes(),
    )
    def test_activity_event_fields_match_inputs(
        self,
        event_type: EventType,
        agent_name: str,
        ticker: str,
        message: str,
        timestamp: datetime,
    ) -> None:
        """All fields on a constructed ActivityEvent match the input values exactly."""
        event = ActivityEvent(
            event_type=event_type,
            agent_name=agent_name,
            ticker=ticker,
            message=message,
            timestamp=timestamp,
        )

        assert event.event_type == event_type
        assert event.agent_name == agent_name
        assert event.ticker == ticker
        assert event.message == message
        assert event.timestamp == timestamp
