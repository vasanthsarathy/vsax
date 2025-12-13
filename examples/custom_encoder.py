"""Custom encoder example.

This example shows how to create custom encoders by subclassing AbstractEncoder.
We'll create a DateEncoder that encodes dates as hypervectors.
"""

from datetime import date
from vsax import create_fhrr_model, VSAMemory, AbstractEncoder, ScalarEncoder


class DateEncoder(AbstractEncoder):
    """Custom encoder for dates.

    Encodes dates by combining encoded year, month, and day
    using role-filler binding.
    """

    def __init__(self, model, memory):
        """Initialize DateEncoder.

        Args:
            model: VSAModel instance.
            memory: VSAMemory instance.
        """
        super().__init__(model, memory)

        # Ensure role symbols exist in memory
        for role in ["year", "month", "day"]:
            if role not in memory:
                memory.add(role)

        # Create scalar encoder for numeric values
        self.scalar_encoder = ScalarEncoder(model, memory)

    def encode(self, date_obj):
        """Encode a date object.

        Args:
            date_obj: Python datetime.date object.

        Returns:
            Encoded hypervector representing the date.
        """
        # Add value symbols to memory if needed
        for symbol in ["year_val", "month_val", "day_val"]:
            if symbol not in self.memory:
                self.memory.add(symbol)

        # Encode each component
        year_hv = self.scalar_encoder.encode("year_val", float(date_obj.year))
        month_hv = self.scalar_encoder.encode("month_val", float(date_obj.month))
        day_hv = self.scalar_encoder.encode("day_val", float(date_obj.day))

        # Bind each value with its role
        year_bound = self.model.opset.bind(
            self.memory["year"].vec,
            year_hv.vec
        )
        month_bound = self.model.opset.bind(
            self.memory["month"].vec,
            month_hv.vec
        )
        day_bound = self.model.opset.bind(
            self.memory["day"].vec,
            day_hv.vec
        )

        # Bundle all role-filler pairs
        result = self.model.opset.bundle(year_bound, month_bound, day_bound)

        return self.model.rep_cls(result)


class ColorEncoder(AbstractEncoder):
    """Custom encoder for RGB colors.

    Encodes colors by binding channel names with their values.
    """

    def __init__(self, model, memory):
        """Initialize ColorEncoder."""
        super().__init__(model, memory)

        # Add channel symbols
        for channel in ["red_channel", "green_channel", "blue_channel"]:
            if channel not in memory:
                memory.add(channel)

        # Scalar encoder for color values (0-255)
        self.scalar_encoder = ScalarEncoder(model, memory, min_val=0, max_val=255)

    def encode(self, rgb_tuple):
        """Encode an RGB color tuple.

        Args:
            rgb_tuple: Tuple of (red, green, blue) values (0-255).

        Returns:
            Encoded hypervector representing the color.
        """
        r, g, b = rgb_tuple

        # Add value symbols
        for symbol in ["r_val", "g_val", "b_val"]:
            if symbol not in self.memory:
                self.memory.add(symbol)

        # Encode each channel
        r_hv = self.scalar_encoder.encode("r_val", float(r))
        g_hv = self.scalar_encoder.encode("g_val", float(g))
        b_hv = self.scalar_encoder.encode("b_val", float(b))

        # Bind with channel names
        r_bound = self.model.opset.bind(self.memory["red_channel"].vec, r_hv.vec)
        g_bound = self.model.opset.bind(self.memory["green_channel"].vec, g_hv.vec)
        b_bound = self.model.opset.bind(self.memory["blue_channel"].vec, b_hv.vec)

        # Bundle all channels
        result = self.model.opset.bundle(r_bound, g_bound, b_bound)

        return self.model.rep_cls(result)


def main():
    """Demonstrate custom encoders."""
    print("=" * 60)
    print("Custom Encoder Examples")
    print("=" * 60)

    # Create model and memory
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)

    # Example 1: DateEncoder
    print("\n1. DateEncoder - Encoding dates...")
    date_encoder = DateEncoder(model, memory)

    birthday = date(1990, 5, 15)
    new_year = date(2025, 1, 1)

    birthday_hv = date_encoder.encode(birthday)
    new_year_hv = date_encoder.encode(new_year)

    print(f"   Encoded: {birthday} → hypervector shape {birthday_hv.vec.shape}")
    print(f"   Encoded: {new_year} → hypervector shape {new_year_hv.vec.shape}")

    # Example 2: ColorEncoder
    print("\n2. ColorEncoder - Encoding RGB colors...")
    color_encoder = ColorEncoder(model, memory)

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    purple = (128, 0, 128)

    red_hv = color_encoder.encode(red)
    green_hv = color_encoder.encode(green)
    purple_hv = color_encoder.encode(purple)

    print(f"   Encoded: RGB{red} (red) → hypervector")
    print(f"   Encoded: RGB{green} (green) → hypervector")
    print(f"   Encoded: RGB{purple} (purple) → hypervector")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Subclass AbstractEncoder to create custom encoders")
    print("  - Implement encode() method with your custom logic")
    print("  - Reuse existing encoders (ScalarEncoder, DictEncoder, etc.)")
    print("  - Combine binding and bundling for complex structures")
    print("=" * 60)


if __name__ == "__main__":
    main()
