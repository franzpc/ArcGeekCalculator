def classFactory(iface):
    from .plugin import ArcGeekCalculator
    return ArcGeekCalculator(iface)