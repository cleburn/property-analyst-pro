"""
Metro configuration loader and utilities.
Handles loading city configs and providing defaults for the Investment Analyzer.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LTRTier:
    """Represents a single LTR rent-to-price tier."""
    max_price: Optional[float]
    rate: float


@dataclass
class MetroConfig:
    """Configuration for a single metro area."""
    key: str
    display_name: str
    state: str
    zillow_city: str
    zillow_metro: str
    has_str_data: bool
    airbnb_file: Optional[str]
    property_tax_rate: float
    insurance_rate: float
    ltr_tiers: List[LTRTier]
    zillow_search_url: str
    property_tax_note: Optional[str] = None
    zillow_cities: Optional[List[str]] = None  # Multiple cities for metro areas with suburbs

    def get_zillow_cities(self) -> List[str]:
        """Get list of Zillow cities to include. Falls back to single zillow_city if not specified."""
        if self.zillow_cities:
            return self.zillow_cities
        return [self.zillow_city]

    def get_ltr_rate(self, price: float) -> float:
        """Get the LTR rent-to-price rate for a given property price."""
        for tier in self.ltr_tiers:
            if tier.max_price is None or price < tier.max_price:
                return tier.rate
        # Fallback to last tier if price exceeds all
        return self.ltr_tiers[-1].rate

    def get_ltr_rate_display(self, price: float) -> str:
        """Get the LTR rent-to-price rate as a display string (e.g., '0.65%')."""
        rate = self.get_ltr_rate(price)
        return f"{rate * 100:.2f}%"

    def calculate_ltr_rent(self, price: float) -> float:
        """Calculate monthly LTR rent for a property price."""
        return price * self.get_ltr_rate(price)


class MetroConfigLoader:
    """Loads and manages metro configurations from YAML."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config/metros.yaml relative to this file
            config_path = Path(__file__).parent / "metros.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._metros: Dict[str, MetroConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Parse metros
        for key, data in self._config.get('metros', {}).items():
            ltr_tiers = [
                LTRTier(
                    max_price=tier.get('max_price'),
                    rate=tier['rate']
                )
                for tier in data.get('ltr_tiers', [])
            ]

            self._metros[key] = MetroConfig(
                key=key,
                display_name=data['display_name'],
                state=data['state'],
                zillow_city=data['zillow_city'],
                zillow_metro=data['zillow_metro'],
                has_str_data=data.get('has_str_data', False),
                airbnb_file=data.get('airbnb_file'),
                property_tax_rate=data['property_tax_rate'],
                insurance_rate=data['insurance_rate'],
                ltr_tiers=ltr_tiers,
                zillow_search_url=data['zillow_search_url'],
                property_tax_note=data.get('property_tax_note'),
                zillow_cities=data.get('zillow_cities')
            )

    @property
    def default_metro(self) -> str:
        """Get the default metro key."""
        return self._config.get('default_metro', 'austin')

    def get_metro(self, key: str) -> MetroConfig:
        """Get configuration for a specific metro."""
        if key not in self._metros:
            raise ValueError(f"Unknown metro: {key}. Available: {list(self._metros.keys())}")
        return self._metros[key]

    def list_metros(self) -> List[str]:
        """List all available metro keys."""
        return list(self._metros.keys())

    def list_metros_by_state(self) -> Dict[str, List[str]]:
        """Group metros by state for UI display."""
        by_state: Dict[str, List[str]] = {}
        for key, metro in self._metros.items():
            state = metro.state
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(key)
        return by_state

    def get_metro_display_options(self) -> Dict[str, str]:
        """Get display name mapping for UI dropdowns. Returns {key: display_name}."""
        return {key: metro.display_name for key, metro in self._metros.items()}

    def get_metros_with_str(self) -> List[str]:
        """Get list of metros that have STR data available."""
        return [key for key, metro in self._metros.items() if metro.has_str_data]

    def get_metros_without_str(self) -> List[str]:
        """Get list of metros that do NOT have STR data (LTR only)."""
        return [key for key, metro in self._metros.items() if not metro.has_str_data]


# Singleton instance for easy import
_loader: Optional[MetroConfigLoader] = None


def get_config_loader() -> MetroConfigLoader:
    """Get the singleton config loader instance."""
    global _loader
    if _loader is None:
        _loader = MetroConfigLoader()
    return _loader


def get_metro_config(metro_key: str) -> MetroConfig:
    """Convenience function to get a metro config."""
    return get_config_loader().get_metro(metro_key)


# Make config directory a proper package
__all__ = [
    'MetroConfig',
    'MetroConfigLoader',
    'LTRTier',
    'get_config_loader',
    'get_metro_config',
]
