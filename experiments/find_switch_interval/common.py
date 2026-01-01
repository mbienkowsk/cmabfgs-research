from enum import Enum


class ObjectiveChoice(Enum):
    ELLIPTIC = "Elliptic"
    RASTRIGIN = "Rastrigin"


class OptimumPosition(Enum):
    MIDDLE = "middle"
    CORNER = "corner"
    OUTSIDE_CORNER = "outside_corner"

    def get_bounds(self):
        match self:
            case OptimumPosition.MIDDLE:
                return (-100.0, 100.0)
            case OptimumPosition.CORNER:
                return (-180.0, 20.0)
            case OptimumPosition.OUTSIDE_CORNER:
                return (-220.0, -20.0)

    def to_plot_label(self):
        bounds = self.get_bounds()
        match self:
            case OptimumPosition.MIDDLE:
                label = "pośrodku obszaru dopuszczalnego"
            case OptimumPosition.CORNER:
                label = "w rogu obszaru dopuszczalnego"
            case OptimumPosition.OUTSIDE_CORNER:
                label = "poza rogiem obszaru dopuszczalnego"
        return f"{label} (granice: {bounds})"
