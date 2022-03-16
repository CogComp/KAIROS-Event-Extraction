#export GRB_LICENSE_FILE=/shared/ccgadmin/demos/KAIROS2020-demo/events/gurobi.lic
#export GRB_LICENSE_FILE=/shared/ccgadmin/demos/KAIROS2020-demo/events/gurobi_client_server.lic
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=0
export GRB_LICENSE_FILE=/shared/ccgadmin/app/gurobi/gurobi_client.lic
while true
do 
	# python backend.py
	python backend_new.py
done

