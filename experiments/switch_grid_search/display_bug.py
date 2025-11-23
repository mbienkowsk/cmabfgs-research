from opfunu.cec_based import cec2017

if __name__ == "__main__":
    fun = cec2017.F92017(ndim=10)
    x = [
        28.65134357093416,
        79.89267547797499,
        55.14337860489706,
        -57.019180636897296,
        -41.90935633243056,
        73.93139548679916,
        -99.03857772878108,
        66.30462801394798,
        65.34292892651949,
        -5.497987795962517,
    ]
    print("\n")
    print(f"Function name: {fun.name}")
    print(f"Opfunu's global minimum of F9: {fun.f_global}")
    print(f"value for given x: {fun.evaluate(x)}")
    x_opt = fun.f_shift
    print(f"opfunu's minimal input: {x_opt}")
    print(f"value for opfunu's minimum input: {fun.evaluate(x_opt)}")
