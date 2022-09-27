# POLICY IMPLICATIONS
# NASH BARGAINING NOT READY

using QuantEcon, Optim, Interpolations, LinearAlgebra,PlotlyJS,Distributions, Pkg, Roots
abstract type Matching
end

# DMP functions
function wv!(av,αv,dmp::Matching)
    r,b,Cf,Cv,τw,τu,A,ϕ = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:ϕ]
    wv = (ϕ *(r+b+ av) * (A - Cf + Cv) + (r + b + αv)*(τw - τu))/((1+ϕ)*r + (1+ϕ)*b + ϕ*av + αv)
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
    r,b,Cf,Cv,τw,τu,A,γ,β,L,κ,ϕ = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:γ],dmp.pars[:β],dmp.pars[:L],dmp.pars[:κ],dmp.pars[:ϕ]
    f(ev) = - Cv +(κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) ) * (A - ((ϕ *(r+b+ ((b * ev)/(L - ev))) * (A - Cf + Cv) + (r + b + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) ))*(τw - τu))/((1+ϕ)*r + (1+ϕ)*b + ϕ*((b * ev)/(L - ev)) + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) ))) - Cf + Cv)/(r + b +(κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) )) 
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
            if zeros == []
                ev = 0.001
                return ev
            else 
                elast_neg = mat_ze[all.(<(0), mat_ze[:,2]), :]
                ev = maximum(elast_neg[:,1])
                return ev[1]
            end
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

####################
# MODEL STRUCT
####################
struct dmp <: Matching
    pars::Dict{Symbol, Float64}
    #employment grid
    E::Vector{Float64}
    # agrid(jE)
    agrid::Vector{Float64}
    # αgrid(jE)
    αgrid::Vector{Float64}
    #wgrid(α,a)
    wgrid::Vector{Float64}
    #rVvgrid(α,a,w)
    rVv::Vector{Float64}
    #(α,a,w)
    rVu::Vector{Float64}
    # rVf
    rVf::Vector{Float64}
    #rVe
    rVe::Vector{Float64}
    # eq matrix
end

#
# MODEL PARAMETERS AND VECTORS
#
function dmp(;τw = 3, τu = 1, b=0.2,Cv = 1, 
    Cf = 1, A = 10, γ = 0.5, 
    κ = 1, β = 0.5, r = 0.02, NE = 10, se=0.001, L = NE + 1e-4, ϕ = 1)
    
    pars = Dict(:τw => τw, :τu => τu, 
    :b => b, :Cv=>Cv, :Cf => Cf, :A => A,
    :γ=> γ, :κ => κ, :β => β, :r => r, :NE => NE,:se =>se, :L=>L, :ϕ =>ϕ)
    E = collect(0.001:se:NE)
    agrid = similar(E)
    αgrid=similar(E)
    wgrid = similar(E)
    rVu = similar(E)
    rVv = similar(E)
    rVf = similar(E)
    rVe = similar(E)
    wgrid = similar(E)
    return dmp(pars, E, agrid, αgrid, wgrid, rVv, rVu,rVf,rVe)
end

function q_iter!(lab_descount,q,new_ne,iter,τv)
    model = dmp(;NE = new_ne, ϕ = τv)
    qgmat = eq_values(model)
    ev = qgmat[1,5]
    a = av!(ev,model)
    α = αv!(ev,model)
    w = wv!(a,α,model)
    rvu_ = rvu!(a,w,model)
    rvu_eq = qgmat[1,2]
    if abs(rvu_ - rvu_eq) > 1e-2
        return print("RVU NO DA LO MISMO")
    end
    if rvu_eq < q
        new_ne *= lab_descount
        iter += 1
        Lab = new_ne
        print("
        L iteration $iter, L = $Lab")
        if Lab < 3e-02
            print("
            WARNING: labour market has shrinked")
        end
        return q_iter!(lab_descount,q,new_ne,iter,τv)
    else 
        iter = 0
        ev = qgmat[1,5]
        αv = qgmat[1,6]
        av = qgmat[1,7]
        wv = qgmat[1,8]  
        # rvv
        rvv = qgmat[1,1]
        # rVu
        rvu = qgmat[1,2]
        # rVe 
        rve = qgmat[1,3]
        # rVf 
        rvf = qgmat[1,4]
        # L'
        lab_force = new_ne
        # self_employ 
        self_employ = 10 - new_ne
        #  unemployment
        uq = new_ne - ev
        # take home wage
        w_net = wv - model.pars[:τw]
        
        return rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net
    end
end

function iter_loop(ft,lab_discount)
    τgrid = collect(1:1:ft)
    q_comparatives_x = zeros(14,size(τgrid,1))
    τ_iter = 0
    iter = 0
    new_ne = 10
    Threads.@threads for jτ in eachindex(τgrid)
        iter = 0;
        τ_iter += 1;
        print("
        τ iterations $τ_iter
        ") 
        τv = τgrid[jτ]
        q = 1 
        rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net = q_iter!(lab_discount,q,new_ne,iter,τv)
        q_comparatives_x[1,jτ] = rvv
        q_comparatives_x[2,jτ] = rvu
        q_comparatives_x[3,jτ] = rve
        q_comparatives_x[4,jτ] = rvf
        q_comparatives_x[5,jτ] = ev
        q_comparatives_x[6,jτ] = αv 
        q_comparatives_x[7,jτ] = av 
        q_comparatives_x[8,jτ] = wv 
        q_comparatives_x[9,jτ] = uq
        q_comparatives_x[10,jτ] = lab_force
        q_comparatives_x[11,jτ] = self_employ
        q_comparatives_x[12,jτ] = w_net
    end
    return q_comparatives_x
end

ϕ_loop = iter_loop(300,0.99)

model = dmp(;);
τgrid = collect(-5:1:5);
 results_alpha = zeros(size(model.E,1)
 , size(τgrid,1));
 results_rvu = zeros(size(model.E,1)
 , size(τgrid,1));
 results_rvv= zeros(size(model.E,1)
 ,size(τgrid,1));
 results_w= zeros(size(model.E,1)
 , size(τgrid,1));
 for jτ in eachindex(τgrid)
    τv = τgrid[jτ]
    model = dmp(;NE = 10, ϕ=τv, A = 10);
    rV!(model);
    results_rvu[:,jτ] = model.rVu;
    results_rvv[:,jτ] = model.rVv;
    results_w[:,jτ] = model.wgrid;
    results_alpha[:,jτ] = model.αgrid
end
plot_rVv = plot([scatter(x= model.E, y=(results_rvv[:,jτ]), 
name = "y = rVv(ϕ = $(τgrid[jτ]))") for jτ in eachindex(τgrid)], 
Layout(title="", xaxis_title="Employment"))

plot_rVu = plot([scatter(x= model.E, y=(results_rvu[:,jτ]), 
name = "y = rVu(ϕ = $(τgrid[jτ]))") for jτ in eachindex(τgrid)], 
Layout(title="", xaxis_title="Employment"))

plot_w = plot([scatter(x= model.E, y=(results_w[:,jτ]), 
name = "y =w(ϕ = $(τgrid[jτ]))") for jτ in eachindex(τgrid)], 
Layout(title="", xaxis_title="Employment"))



# LEVEL

τgrid = collect(1:1:300)
self_employment_low_ϕ = 10 .- (ϕ_loop[5,:] .+ ϕ_loop[9,:])
ϕ_loop[11,:]
employment_ϕ_low = plot([scatter(x= τgrid, y=ϕ_loop[5,:], 
name = "E*(B)")], 
Layout(title="Employment level", xaxis_title="ϕ"));

unemployment_ϕ_low = plot([scatter(x= τgrid, y=ϕ_loop[9,:], 
    name = "U*(B)")], 
Layout(title="Unemployment level", xaxis_title="ϕ"));

self_ϕ_low = plot([scatter(x= τgrid, y=self_employment_low_ϕ, 
name = "H*(B)")], 
Layout(title="Self employment level", xaxis_title="ϕ"));

pop_plot_low_ϕ = ([employment_ϕ_low unemployment_ϕ_low self_ϕ_low])
savefig(pop_plot_low_ϕ,"replication_files/new_figures/NASH/level_new.png")


# RATES

tot_pop_low_ϕ = self_employment_low_ϕ .+ ϕ_loop[5,:] .+ ϕ_loop[9,:]
tot_pop_low_ϕ = ϕ_loop[11,:] .+ ϕ_loop[5,:] .+ ϕ_loop[9,:]
e_rate_low = ϕ_loop[5,:] ./ tot_pop_low_ϕ;
u_rate_low = ϕ_loop[9,:] ./ tot_pop_low_ϕ;
self_rate_low = ϕ_loop[11,:] ./ tot_pop_low_ϕ
plot_rates_low = plot([scatter(x=τgrid, 
y = e_rate_low, name="E/L"); scatter(x=τgrid, 
y=u_rate_low, name="U/L"); scatter(x=τgrid, 
y=self_rate_low, name="H/L")],
 Layout(title = "Employment and Unemployment rates, q = 1",xaxis_title = "ϕ"))

savefig(plot_rates_low, "replication_files/new_figures/NASH/rates_new.png")


# WAGE

wage = plot([scatter(x= τgrid, y=ϕ_loop[8,:], 
name = "w(ϕ)")], 
Layout(title="Wage", xaxis_title="ϕ"));
take_home_wage = ϕ_loop[8,:] .- 3 
wage_at_home = plot([scatter(x= τgrid, y=take_home_wage, 
name = "W*(ϕ) - τw")], 
Layout(title="Take home wage", xaxis_title="ϕ"));

wage_tot = [wage wage_at_home]
savefig(wage_tot,"replication_files/new_figures/NASH/wages_new.png")

# RVU 

unemployment_ϕ_low = plot([scatter(x= τgrid, y=ϕ_loop[2,:], 
    name = "rVu*(ϕ)")], 
Layout(title="rVu", xaxis_title="ϕ"))
savefig(unemployment_ϕ_low,"replication_files/new_figures/NASH/rvu_new.png")





