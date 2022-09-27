using Roots

# DMP functions

function wv!(av,αv,dmp::Matching)
    r,b,Cf,Cv,τw,τu,A = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A]
    wv = ((r+b+ av) * (A - Cf + Cv) + (r + b + αv)*(τw - τu))/(2*r + 2*b + av + αv)
    return wv
end

function av!(ev,dmp::Matching)
    b,L = dmp.pars[:b],dmp.pars[:L]
    av = (b * ev)/(L - ev)
    # a = b*E/U
    return av
end

function αv!(ev,dmp::Matching)
    κ,γ,b,β,L  = dmp.pars[:κ],dmp.pars[:γ],dmp.pars[:b],dmp.pars[:β],dmp.pars[:L]
    αv = κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))  
    return αv
end


function rvv!(αv,wv,dmp::Matching)
    Cv,A,Cf,r,b = dmp.pars[:Cv],dmp.pars[:A],dmp.pars[:Cf],dmp.pars[:r],dmp.pars[:b]
    rvv = - Cv + αv * (A - wv - Cf + Cv)/(r + b +αv) 
    return rvv
end


function rV!(dmp::Matching) 
    for je in eachindex(dmp.E)
        # fill the vectors
        ev = dmp.E[je] 
        av = av!(ev,dmp)
        αv = αv!(ev,dmp)
        wv = wv!(av,αv,dmp)
        # Value functions
        rvv = rvv!(αv,wv,dmp)
        rvf = rvf!(αv,wv,dmp)
        rve = rve!(av,wv,dmp)
        rvu = rvu!(av,wv,dmp)
        # Complete the vectors for each je
        dmp.rVv[je] = rvv
        dmp.rVu[je] = rvu
        dmp.rVf[je] = rvf
        dmp.rVe[je] = rve
        dmp.wgrid[je] = wv
        dmp.agrid[je] = av
        dmp.αgrid[je] = αv
    end
end




function rvv_zeros(dmp)
    # this function takes rvv as a function of E and parameters
    r,b,Cf,Cv,τw,τu,A,γ,β,L,κ = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:γ],dmp.pars[:β],dmp.pars[:L],dmp.pars[:κ]
    f(ev) = - Cv + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))) * (A - ( ((r+b+ ((b * ev)/(L - ev))) * (A - Cf + Cv) + (r + b + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))))*(τw - τu))/(2*r + 2*b + ((b * ev)/(L - ev)) + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))))) - Cf + Cv)/(r + b +(κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))))
    # then calculate the zeros for the function in an interval minimum(E) to L
    # then 
    zeros = find_zeros(f,minimum(dmp.E),dmp.pars[:L])
    return zeros
end

function zeros_elast(zeros,dmp)
    # calculate the zeros
    mat =ones(length(zeros),2)
    mat[:,1] = zeros[:,1]
    # for each zero I will calculate the derivatives
    for jz in eachindex(zeros)
        zv = zeros[jz]
        mat[jz,2] = rvvp!(zv,0.0001,dmp)
    end
    elast =mat[:,2]
    return elast
end

function selection!(dmp)
    # this function returns an employment in equilibrium once some conditions are satisfied
    # i) if the rvv is entirely < 0 then the eq is 0
    # ii) if it is positive look for the zeros, and return a positive value only if it is unique and got rvv'e* <0 
    # iii) if it is unique and rvv'e > 0 return 0
    # iv) if it is not unique take the equilibriums with negative rvv'e*. return the maximum
    # check if the vector is positive (check that it is not negative)
    rvv_posit = filter(x -> x>=0,dmp.rVv)
    if rvv_posit != []
        # Find the equilibrium
        zeros = rvv_zeros(dmp)
        # calculate the derivative for all the vector E 
        elast = zeros_elast(zeros,dmp)
        mat_ze = [zeros elast]
        # check if it is unique 
        # if derivative < 0 return the equilibrium
        if length(zeros) == 1
            if elast[1,1] < 0
                ev = zeros
                return ev
            else 
                ev = minimum(dmp.E)
                return ev[1]
            end
        else 
            elast_neg = mat_ze[all.(<(0), mat_ze[:,2]), :]
            ev = maximum(elast_neg[:,1])
            return ev[1]
        end
    else 
        if dmp.E == []
            ev = 0.001
            return ev
        else 
            ev = minimum(dmp.E)
        return ev[1]
        end
    end
end

function eq_values(dmp)
        # rV! es una función que rellena los vectores vacíos. Entonces le damos un modelo DMP 
    # Ese modelo tiene vectores vacíos de longitud E(vector de estado) que rV! llena
    # para cada E te devuelve un a,α,wv,rvv 
    gmat = zeros(1,10)
    rV!(dmp)
    # take the equilibrium at employment
    ev = selection!(dmp)
    ev = ev[1]
    # evaluate all functions at equilibrium
    av = av!(ev,dmp)
    αv = αv!(ev,dmp)
    wv = wv!(av,αv,dmp)
    # Value functions
    rvv = rvv!(αv,wv,dmp)
    rvf = rvf!(αv,wv,dmp)
    rve = rve!(av,wv,dmp)
    rvu = rvu!(av,wv,dmp)
    # filling the equilibrium matrix
    gmat[1,1] = rvv
    # rVu
    gmat[1,2] = rvu
    # rVe 
    gmat[1,3] = rve
    # rVf 
    gmat[1,4] = rvf
    # Employment 
    gmat[1,5] = ev
    # αv 
    gmat[1,6] = αv
    # av
    gmat[1,7] = av
    # wv 
    gmat[1,8] = wv
    # unemployment 
    gmat[1,9] = dmp.pars[:L] - ev
    return gmat 
end


function rvu!(av,wv,dmp::Matching)
    τu,τw,r,b = dmp.pars[:τu],dmp.pars[:τw],dmp.pars[:r],dmp.pars[:b]
    rvu = - τu + av * (wv - τw + τu)/(r + b + av)
    return rvu
end

function rvf!(αv,wv,dmp::Matching)
    A,Cf,b,Cv,r = dmp.pars[:A],dmp.pars[:Cf],dmp.pars[:b],dmp.pars[:Cv],dmp.pars[:r]
    rvf = A - wv - Cf - b*((A - wv + Cv - Cf)/(r + b + αv)) 
    return rvf
end

function rve!(av,wv,dmp::Matching)
    τw,τu,b,r = dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:b],dmp.pars[:r]
   rve = wv - τw - b*((wv + τu - τw)/(av + b + r))
   return rve 
end

function rvv_h!(ev,h,dmp::Matching)
    # given an employment level and a 'step' this function returns
    # the value of posting a vacancy evaluated at ev+h
    Cv,A,Cf,r,b = dmp.pars[:Cv],dmp.pars[:A],dmp.pars[:Cf],dmp.pars[:r],dmp.pars[:b]
    evh = ev + h
    αv = αv!(evh,dmp)
    av = av!(evh,dmp)
    wv = wv!(av,αv,dmp)
    rvv = - Cv + αv * (A - wv - Cf + Cv)/(r + b +αv)
    return rvv
end

function rvvp!(ev,h,dmp::Matching)
    # given an empleyment level this function calculates the value of posting a vanacy at ev (rvv)
    # and the value of posting a vavancy at ev+h 
    # Then, this function returns 
    #= 
    rvv'e = (rvv(e+h)-rvv(e))/h
    =#
    αv = αv!(ev,dmp)
    av = av!(ev,dmp)
    wv = wv!(av,αv,dmp)
    rvvh = rvv_h!(ev,h,dmp)
    rvv = rvv!(αv,wv,dmp)
    rvvp = (rvvh - rvv)/h
    return rvvp
end







