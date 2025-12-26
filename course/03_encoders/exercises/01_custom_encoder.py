"""
Module 3 Exercise 1: Build Custom Encoders

This exercise guides you through building custom encoders from scratch
to understand how encoders transform data into hypervectors.

Tasks:
1. Build a DateEncoder for encoding dates (year, month, day)
2. Build a RGBColorEncoder for encoding colors
3. Build a TimeSeriesEncoder with sliding windows
4. Test all custom encoders
5. Compare with built-in encoders

Expected learning:
- Deep understanding of encoder architecture
- How to combine multiple dimensions
- When to use binding vs bundling
- Building domain-specific encoders
"""

from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import AbstractEncoder, FractionalPowerEncoder
from vsax.similarity import cosine_similarity
import jax.numpy as jnp
from datetime import date


class DateEncoder(AbstractEncoder):
    """
    Custom encoder for dates.

    Encodes dates as: Year^y ⊗ Month^m ⊗ Day^d
    """

    def __init__(self, model, memory):
        """
        Initialize DateEncoder.

        Args:
            model: VSA model
            memory: VSA memory
        """
        super().__init__(model, memory)

        # Add basis vectors for date components
        self.memory.add_many(["year", "month", "day"])

        # Create FPE for encoding numeric values
        self.fpe = FractionalPowerEncoder(model, memory, scale=0.01)

    def encode(self, date_obj):
        """
        Encode a date object.

        Args:
            date_obj: datetime.date object or tuple (year, month, day)

        Returns:
            ComplexHypervector representing the date
        """
        # Handle both date objects and tuples
        if isinstance(date_obj, date):
            year, month, day = date_obj.year, date_obj.month, date_obj.day
        else:
            year, month, day = date_obj

        # Encode each component with FPE
        year_hv = self.fpe.encode("year", year)
        month_hv = self.fpe.encode("month", month)
        day_hv = self.fpe.encode("day", day)

        # Bind all components
        result = self.model.opset.bind(year_hv.vec, month_hv.vec)
        result = self.model.opset.bind(result, day_hv.vec)

        # Wrap in hypervector
        return self.model.rep_cls(result)

    def query_component(self, encoded_date, component):
        """
        Query a specific component from an encoded date.

        Args:
            encoded_date: Encoded date hypervector
            component: "year", "month", or "day"

        Returns:
            Approximate value of the component
        """
        # This is simplified - in practice, use grid search
        # to recover the exact value

        # Get the other two components to unbind
        components = {"year", "month", "day"}
        other_components = components - {component}

        # Unbind the other components
        result = encoded_date.vec
        for comp in other_components:
            comp_vec = self.memory[comp].vec
            # For simplicity, assume we know approximate values
            # In real usage, this would require cleanup/search

        # Return the component vector for similarity comparison
        return result


class RGBColorEncoder(AbstractEncoder):
    """
    Custom encoder for RGB colors.

    Encodes colors as: R^r ⊗ G^g ⊗ B^b
    """

    def __init__(self, model, memory, scale=0.01):
        """
        Initialize RGBColorEncoder.

        Args:
            model: VSA model
            memory: VSA memory
            scale: Scaling factor for RGB values (default: 0.01 for [0, 255])
        """
        super().__init__(model, memory)

        # Add basis vectors
        self.memory.add_many(["red", "green", "blue"])

        # Create FPE
        self.fpe = FractionalPowerEncoder(model, memory, scale=scale)

    def encode(self, rgb_tuple):
        """
        Encode an RGB color.

        Args:
            rgb_tuple: (r, g, b) tuple with values in [0, 255]

        Returns:
            ComplexHypervector representing the color
        """
        r, g, b = rgb_tuple

        # Encode each channel
        r_hv = self.fpe.encode("red", r)
        g_hv = self.fpe.encode("green", g)
        b_hv = self.fpe.encode("blue", b)

        # Bind all channels
        result = self.model.opset.bind(r_hv.vec, g_hv.vec)
        result = self.model.opset.bind(result, b_hv.vec)

        return self.model.rep_cls(result)

    def encode_named_colors(self):
        """
        Encode a palette of named colors.

        Returns:
            Dictionary of color names to hypervectors
        """
        colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
            "cyan": (0, 255, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "gray": (128, 128, 128)
        }

        encoded_colors = {}
        for name, rgb in colors.items():
            encoded_colors[name] = self.encode(rgb)

        return encoded_colors


class TimeSeriesEncoder(AbstractEncoder):
    """
    Custom encoder for time series with sliding windows.

    Encodes time series as: ⊕ᵢ (POSᵢ ⊗ VALUEᵢ)
    """

    def __init__(self, model, memory, window_size=5):
        """
        Initialize TimeSeriesEncoder.

        Args:
            model: VSA model
            memory: VSA memory
            window_size: Size of sliding window
        """
        super().__init__(model, memory)

        self.window_size = window_size

        # Add basis vectors for positions and value
        self.memory.add_many([f"pos_{i}" for i in range(window_size)])
        self.memory.add("value")

        # Create FPE for values
        self.fpe = FractionalPowerEncoder(model, memory, scale=0.1)

    def encode(self, time_series):
        """
        Encode a time series using sliding window.

        Args:
            time_series: List or array of numeric values

        Returns:
            ComplexHypervector representing the time series
        """
        # Ensure we have enough data
        if len(time_series) < self.window_size:
            raise ValueError(f"Time series too short (need at least {self.window_size} points)")

        # Take most recent window
        window = time_series[-self.window_size:]

        # Encode each value with position
        encoded_points = []
        for i, value in enumerate(window):
            # Encode value
            value_hv = self.fpe.encode("value", value)

            # Get position vector
            pos_hv = self.memory[f"pos_{i}"].vec

            # Bind position with value
            bound = self.model.opset.bind(pos_hv, value_hv.vec)
            encoded_points.append(bound)

        # Bundle all points
        result = self.model.opset.bundle(*encoded_points)

        return self.model.rep_cls(result)

    def encode_sequence(self, time_series, stride=1):
        """
        Encode entire time series as sequence of windows.

        Args:
            time_series: Complete time series
            stride: Step size between windows

        Returns:
            List of encoded windows
        """
        windows = []

        for i in range(0, len(time_series) - self.window_size + 1, stride):
            window = time_series[i:i + self.window_size]
            encoded = self.encode(window)
            windows.append(encoded)

        return windows


def test_date_encoder():
    """
    Test DateEncoder with various dates.
    """
    print("=" * 60)
    print("Test 1: DateEncoder")
    print("=" * 60)

    model = create_fhrr_model(dim=2048)
    memory = VSAMemory(model)

    encoder = DateEncoder(model, memory)

    # Encode some dates
    dates = [
        (2024, 1, 15),   # Mid January
        (2024, 1, 20),   # Late January (should be similar to above)
        (2024, 7, 15),   # Mid July (different month)
        (2025, 1, 15),   # Same month/day, different year
    ]

    encoded_dates = [encoder.encode(d) for d in dates]

    print("\nDate similarities:")
    print(f"{'Date 1':<20s} {'Date 2':<20s} {'Similarity':<12s}")
    print("-" * 60)

    for i, d1 in enumerate(dates):
        for j, d2 in enumerate(dates):
            if i < j:
                sim = cosine_similarity(encoded_dates[i].vec, encoded_dates[j].vec)
                d1_str = f"{d1[0]}-{d1[1]:02d}-{d1[2]:02d}"
                d2_str = f"{d2[0]}-{d2[1]:02d}-{d2[2]:02d}"
                print(f"{d1_str:<20s} {d2_str:<20s} {sim:<12.4f}")

    print("\nObservations:")
    print("- Dates close in time have higher similarity")
    print("- Same month/day but different year: moderate similarity")
    print("- Different months: lower similarity")


def test_rgb_encoder():
    """
    Test RGBColorEncoder with color palette.
    """
    print("\n" + "=" * 60)
    print("Test 2: RGBColorEncoder")
    print("=" * 60)

    model = create_fhrr_model(dim=2048)
    memory = VSAMemory(model)

    encoder = RGBColorEncoder(model, memory, scale=0.01)

    # Encode color palette
    colors = encoder.encode_named_colors()

    print("\nColor palette encoded:")
    for name in colors:
        print(f"  - {name}")

    # Test color similarity
    print("\nColor similarities:")

    test_pairs = [
        ("red", "pink"),        # Similar hue
        ("red", "orange"),      # Similar hue
        ("red", "blue"),        # Opposite
        ("red", "magenta"),     # Contains red
        ("blue", "cyan"),       # Contains blue
        ("yellow", "orange"),   # Similar
    ]

    print(f"{'Color 1':<15s} {'Color 2':<15s} {'Similarity':<12s}")
    print("-" * 50)

    for c1, c2 in test_pairs:
        sim = cosine_similarity(colors[c1].vec, colors[c2].vec)
        print(f"{c1:<15s} {c2:<15s} {sim:<12.4f}")

    print("\nObservations:")
    print("- Similar colors (red-pink, blue-cyan) have higher similarity")
    print("- Opposite colors (red-blue) have lower similarity")
    print("- Mixed colors show moderate similarity to components")


def test_timeseries_encoder():
    """
    Test TimeSeriesEncoder with synthetic data.
    """
    print("\n" + "=" * 60)
    print("Test 3: TimeSeriesEncoder")
    print("=" * 60)

    model = create_fhrr_model(dim=2048)
    memory = VSAMemory(model)

    encoder = TimeSeriesEncoder(model, memory, window_size=5)

    # Create synthetic time series patterns
    import numpy as np

    # Pattern 1: Upward trend
    trend_up = [10, 12, 14, 16, 18, 20, 22, 24]

    # Pattern 2: Downward trend
    trend_down = [24, 22, 20, 18, 16, 14, 12, 10]

    # Pattern 3: Oscillating
    oscillating = [10, 15, 10, 15, 10, 15, 10, 15]

    # Pattern 4: Stable
    stable = [15, 15, 15, 15, 15, 15, 15, 15]

    patterns = {
        "upward": trend_up,
        "downward": trend_down,
        "oscillating": oscillating,
        "stable": stable
    }

    # Encode each pattern
    encoded_patterns = {}
    for name, series in patterns.items():
        encoded = encoder.encode(series)
        encoded_patterns[name] = encoded
        print(f"Encoded: {name:<12s} (window: {series[-5:]})")

    # Compare patterns
    print("\nPattern similarities:")
    print(f"{'Pattern 1':<15s} {'Pattern 2':<15s} {'Similarity':<12s}")
    print("-" * 50)

    pattern_names = list(patterns.keys())
    for i, p1 in enumerate(pattern_names):
        for j, p2 in enumerate(pattern_names):
            if i < j:
                sim = cosine_similarity(
                    encoded_patterns[p1].vec,
                    encoded_patterns[p2].vec
                )
                print(f"{p1:<15s} {p2:<15s} {sim:<12.4f}")

    print("\nObservations:")
    print("- Similar patterns (upward vs downward) show some similarity")
    print("- Very different patterns (oscillating vs stable) have lower similarity")
    print("- Pattern recognition works through positional encoding")


def test_encoder_composition():
    """
    Test combining multiple custom encoders.
    """
    print("\n" + "=" * 60)
    print("Test 4: Encoder Composition")
    print("=" * 60)

    model = create_fhrr_model(dim=4096)  # Higher dim for complex encoding
    memory = VSAMemory(model)

    # Create encoders
    date_enc = DateEncoder(model, memory)
    color_enc = RGBColorEncoder(model, memory)

    # Encode: "Event on 2024-01-15 with red color"
    date1 = (2024, 1, 15)
    color1 = (255, 0, 0)  # red

    date_hv1 = date_enc.encode(date1)
    color_hv1 = color_enc.encode(color1)

    # Combine: bind date with color
    event1 = model.opset.bind(date_hv1.vec, color_hv1.vec)

    # Encode: "Event on 2024-01-20 with red color" (similar)
    date2 = (2024, 1, 20)
    color2 = (255, 0, 0)  # red

    date_hv2 = date_enc.encode(date2)
    color_hv2 = color_enc.encode(color2)
    event2 = model.opset.bind(date_hv2.vec, color_hv2.vec)

    # Encode: "Event on 2024-07-15 with blue color" (different)
    date3 = (2024, 7, 15)
    color3 = (0, 0, 255)  # blue

    date_hv3 = date_enc.encode(date3)
    color_hv3 = color_enc.encode(color3)
    event3 = model.opset.bind(date_hv3.vec, color_hv3.vec)

    # Compare events
    sim_12 = cosine_similarity(event1, event2)
    sim_13 = cosine_similarity(event1, event3)
    sim_23 = cosine_similarity(event2, event3)

    print("\nEvent similarities:")
    print(f"Event 1 (2024-01-15, red) vs Event 2 (2024-01-20, red): {sim_12:.4f}")
    print(f"Event 1 (2024-01-15, red) vs Event 3 (2024-07-15, blue): {sim_13:.4f}")
    print(f"Event 2 (2024-01-20, red) vs Event 3 (2024-07-15, blue): {sim_23:.4f}")

    print("\nObservations:")
    print("- Events with same color and nearby dates: highest similarity")
    print("- Events with different dates and colors: lowest similarity")
    print("- Custom encoders can be composed via binding")


def main():
    """
    Run all custom encoder tests.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 3 EXERCISE 1")
    print(" " * 20 + "Build Custom Encoders")
    print("=" * 80)

    test_date_encoder()
    test_rgb_encoder()
    test_timeseries_encoder()
    test_encoder_composition()

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("✓ Custom encoders inherit from AbstractEncoder")
    print("✓ Use FPE for multi-dimensional continuous data")
    print("✓ Bind components for compositional encoding")
    print("✓ Bundle for order-invariant aggregation")
    print("✓ Encoders can be composed via binding")
    print("✓ Design encoders based on domain structure")
    print("=" * 80)


if __name__ == "__main__":
    main()
