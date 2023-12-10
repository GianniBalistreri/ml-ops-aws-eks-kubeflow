
add-user-dex:
	kubectl apply -f add_user.yaml
	kubectl rollout restart deployment dex -n auth

add-profile:
	kubectl apply -f add_profile.yaml

get-profile:
	kubectl get profile

delete-profile:
	kubectl delete profile <username>
	#kubectl delete ns <username>
