
https://www.researchgate.net/publication/317228117_Problem_Definitions_and_Evaluation_Criteria_for_the_CEC_2017_Competition_and_Special_Session_on_Constrained_Single_Objective_Real-Parameter_Optimization
* stronka biedrzyckiego


* wybrac 10 funkcji, zeby nie przegladac 30
* wyglada na to, ze chcemy to robic na "peaku" wsp.uwarunkowania, ktory sie ustali/spadnie, moze brak wzrostu

---

## uruchomic znowu z budzetem cec 10_000 * dim

* wektory startowe dla l-bfgs-b jako najwieksze wektory wlasne macierzy kowariancji

* dlaczego trzeba dodawac transpozycje i /2?
* jak wyglada to gdy nie damy bfgsowi na start macierzy?
* można wrócić do linesearchu, jak w porównaniu do puszczania bfgsa


## todo po kolei
* porównanie performance z maina vs z rewritea - wykresy powinny być +- identyczne, przynajmniej dawać te same wyniki
* ioh? -> porównanie z uruchomieniem bez macierzy

* jest jakiś bug do poprawienia w CEC29
## pytania
* Jak interpolować dane? Tworzymy siatkę co n do wykreślania, czy suma logiczna indeksów?


## bugs
* eigenvalues did not converge - cec 25 10dim seed 5
