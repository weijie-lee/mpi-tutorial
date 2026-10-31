.PHONY: verify clean

verify:
	bash scripts/verify.sh

clean:
	rm -rf build
	$(MAKE) -C ch04-hardware/04-hardware clean
	$(MAKE) -C ch08-rdma-verbs/08-rdma-verbs clean
