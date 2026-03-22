const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Autonet", function () {
    let deployer, user1, user2, backer1, backer2;
    let autonet, autonetFactory;
    let economy, registry, repToken, timelock;
    let mockProject;

    // Deploy a minimal jurisdiction infrastructure for testing
    async function deployMockJurisdiction() {
        // Deploy mock contracts that implement the minimal interfaces we need
        const MockRegistry = await ethers.getContractFactory("AutonetMockRegistry");
        const mockRegistry = await MockRegistry.deploy();

        const MockRepToken = await ethers.getContractFactory("AutonetMockRepToken");
        const mockRepToken = await MockRepToken.deploy("Test Rep", "REP");

        // For timelock and economy, use simple addresses (we won't call their functions in basic tests)
        const mockEconomy = deployer.address; // placeholder
        const mockTimelock = deployer.address; // deployer acts as timelock for admin actions

        return {
            economy: mockEconomy,
            registry: await mockRegistry.getAddress(),
            repToken: await mockRepToken.getAddress(),
            timelock: mockTimelock,
            registryContract: mockRegistry,
            repTokenContract: mockRepToken,
        };
    }

    beforeEach(async function () {
        [deployer, user1, user2, backer1, backer2] = await ethers.getSigners();

        // Deploy mock jurisdiction
        const jurisdiction = await deployMockJurisdiction();
        economy = jurisdiction.economy;
        registry = jurisdiction.registry;
        repToken = jurisdiction.repToken;
        timelock = jurisdiction.timelock;

        // Deploy Autonet
        const Autonet = await ethers.getContractFactory("Autonet");
        autonet = await Autonet.deploy(
            economy,
            registry,
            repToken,
            timelock,
            "Autonet Credits - Test",
            "ATN-TEST"
        );

        // Deploy a mock project for service registration
        mockProject = user1.address; // placeholder for project contract
    });

    describe("Deployment", function () {
        it("should deploy with correct parameters", async function () {
            expect(await autonet.name()).to.equal("Autonet Credits - Test");
            expect(await autonet.symbol()).to.equal("ATN-TEST");
            expect(await autonet.economyAddress()).to.equal(economy);
            expect(await autonet.registryAddress()).to.equal(registry);
            expect(await autonet.repTokenAddress()).to.equal(repToken);
            expect(await autonet.timelockAddress()).to.equal(timelock);
            expect(await autonet.admin()).to.equal(deployer.address);
            expect(await autonet.currentEpoch()).to.equal(0);
        });

        it("should have correct decimals", async function () {
            expect(await autonet.decimals()).to.equal(18);
        });

        it("should start with zero supply", async function () {
            expect(await autonet.totalSupply()).to.equal(0);
        });
    });

    describe("Service Registration", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("test-service-1"));
        const codebaseHash = "abc123";

        it("should register a general service", async function () {
            await expect(autonet.connect(user1).registerService(serviceId, mockProject, codebaseHash))
                .to.emit(autonet, "ServiceRegistered")
                .withArgs(serviceId, mockProject, await autonet.SERVICE_TYPE_GENERAL(), user1.address);

            const service = await autonet.getService(serviceId);
            expect(service.projectContract).to.equal(mockProject);
            expect(service.codebaseHash).to.equal(codebaseHash);
            expect(service.serviceType).to.equal(await autonet.SERVICE_TYPE_GENERAL());
            expect(service.isActive).to.equal(false);
        });

        it("should register an inference provider", async function () {
            const inferenceEndpoint = user2.address; // placeholder for IInferenceProvider

            await expect(autonet.connect(user1).registerInferenceProvider(
                serviceId, mockProject, inferenceEndpoint, codebaseHash
            ))
                .to.emit(autonet, "ServiceRegistered")
                .withArgs(serviceId, mockProject, await autonet.SERVICE_TYPE_INFERENCE_PROVIDER(), user1.address);

            const service = await autonet.getService(serviceId);
            expect(service.serviceEndpoint).to.equal(inferenceEndpoint);
            expect(service.serviceType).to.equal(await autonet.SERVICE_TYPE_INFERENCE_PROVIDER());
        });

        it("should not allow duplicate service IDs", async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, codebaseHash);

            await expect(autonet.connect(user2).registerService(serviceId, user2.address, "xyz"))
                .to.be.revertedWithCustomError(autonet, "ServiceAlreadyExists");
        });

        it("should track services by creator", async function () {
            const serviceId2 = ethers.keccak256(ethers.toUtf8Bytes("test-service-2"));

            await autonet.connect(user1).registerService(serviceId, mockProject, codebaseHash);
            await autonet.connect(user1).registerService(serviceId2, mockProject, "def456");

            const services = await autonet.getServicesByCreator(user1.address);
            expect(services.length).to.equal(2);
            expect(services[0]).to.equal(serviceId);
            expect(services[1]).to.equal(serviceId2);
        });
    });

    describe("Service Activation", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("activation-test"));

        beforeEach(async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
        });

        it("should allow admin to activate service", async function () {
            await expect(autonet.activateService(serviceId))
                .to.emit(autonet, "ServiceActivated")
                .withArgs(serviceId);

            const service = await autonet.getService(serviceId);
            expect(service.isActive).to.equal(true);
        });

        it("should allow admin to deactivate service", async function () {
            await autonet.activateService(serviceId);

            await expect(autonet.deactivateService(serviceId))
                .to.emit(autonet, "ServiceDeactivated")
                .withArgs(serviceId);

            const service = await autonet.getService(serviceId);
            expect(service.isActive).to.equal(false);
        });

        it("should not allow non-admin to activate", async function () {
            await expect(autonet.connect(user1).activateService(serviceId))
                .to.be.revertedWithCustomError(autonet, "NotAuthorized");
        });
    });

    describe("Usage Attestation", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("usage-test"));

        beforeEach(async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
            await autonet.activateService(serviceId);
        });

        it("should attest usage for active service", async function () {
            const units = 100;

            await expect(autonet.connect(user1).attestUsage(serviceId, units))
                .to.emit(autonet, "UsageAttested")
                .withArgs(serviceId, user1.address, units, 0);

            const service = await autonet.getService(serviceId);
            expect(service.totalUsageUnits).to.equal(units);
        });

        it("should not attest for inactive service", async function () {
            await autonet.deactivateService(serviceId);

            await expect(autonet.connect(user1).attestUsage(serviceId, 100))
                .to.be.revertedWithCustomError(autonet, "ServiceNotActive");
        });

        it("should accumulate usage across attestations", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 100);
            await autonet.connect(user2).attestUsage(serviceId, 50);

            const service = await autonet.getService(serviceId);
            expect(service.totalUsageUnits).to.equal(150);
        });
    });

    describe("Epoch Management", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("epoch-test"));
        const budget = ethers.parseEther("1000");

        beforeEach(async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
            await autonet.activateService(serviceId);
        });

        it("should start new epoch", async function () {
            await expect(autonet.startEpoch(budget))
                .to.emit(autonet, "EpochStarted")
                .withArgs(1, budget);

            expect(await autonet.currentEpoch()).to.equal(1);
        });

        it("should finalize epoch and distribute rewards", async function () {
            await autonet.startEpoch(budget);
            await autonet.connect(user1).attestUsage(serviceId, 100);

            await expect(autonet.finalizeCurrentEpoch())
                .to.emit(autonet, "EpochFinalized");

            const epochStats = await autonet.getEpochStats(1);
            expect(epochStats.totalUsage).to.equal(100);
            expect(epochStats.finalized).to.equal(true);
        });

        it("should distribute proportional rewards to services", async function () {
            const serviceId2 = ethers.keccak256(ethers.toUtf8Bytes("epoch-test-2"));
            await autonet.connect(user2).registerService(serviceId2, user2.address, "hash2");
            await autonet.activateService(serviceId2);

            await autonet.startEpoch(budget);
            await autonet.connect(user1).attestUsage(serviceId, 75);  // 75%
            await autonet.connect(user2).attestUsage(serviceId2, 25); // 25%

            await autonet.finalizeCurrentEpoch();

            const reward1 = await autonet.getServiceEpochReward(serviceId, 1);
            const reward2 = await autonet.getServiceEpochReward(serviceId2, 1);

            // Should be approximately 750 and 250 ATN
            expect(reward1.totalReward).to.equal(ethers.parseEther("750"));
            expect(reward2.totalReward).to.equal(ethers.parseEther("250"));
        });
    });

    describe("Emission Decay", function () {
        it("should configure emission and start first epoch automatically", async function () {
            const initialBudget = ethers.parseEther("10000");
            const decayRate = 9900; // 99% retention = 1% decay
            const duration = 86400; // 1 day

            await expect(autonet.configureEmission(initialBudget, decayRate, duration))
                .to.emit(autonet, "EmissionConfigured")
                .withArgs(initialBudget, decayRate, duration);

            // First epoch should be started automatically
            expect(await autonet.currentEpoch()).to.equal(1);
            expect(await autonet.autoEpochEnabled()).to.equal(true);
            expect(await autonet.epochBudget(1)).to.equal(initialBudget);
        });

        it("should compute decaying budgets correctly", async function () {
            const initialBudget = ethers.parseEther("10000");
            await autonet.configureEmission(initialBudget, 9900, 86400);

            // Epoch 1: 10000
            expect(await autonet.computeEpochBudget(1)).to.equal(ethers.parseEther("10000"));

            // Epoch 2: 10000 * 0.99 = 9900
            expect(await autonet.computeEpochBudget(2)).to.equal(ethers.parseEther("9900"));

            // Epoch 3: 9900 * 0.99 = 9801
            expect(await autonet.computeEpochBudget(3)).to.equal(ethers.parseEther("9801"));

            // Epoch 0: should be 0
            expect(await autonet.computeEpochBudget(0)).to.equal(0);
        });

        it("should compute budget with 50% decay (halving)", async function () {
            const initialBudget = ethers.parseEther("10000");
            await autonet.configureEmission(initialBudget, 5000, 86400);

            expect(await autonet.computeEpochBudget(1)).to.equal(ethers.parseEther("10000"));
            expect(await autonet.computeEpochBudget(2)).to.equal(ethers.parseEther("5000"));
            expect(await autonet.computeEpochBudget(3)).to.equal(ethers.parseEther("2500"));
            expect(await autonet.computeEpochBudget(4)).to.equal(ethers.parseEther("1250"));
        });

        it("should reject invalid emission config", async function () {
            await expect(autonet.configureEmission(0, 9900, 86400))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");

            await expect(autonet.configureEmission(ethers.parseEther("10000"), 0, 86400))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");

            await expect(autonet.configureEmission(ethers.parseEther("10000"), 10001, 86400))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");

            await expect(autonet.configureEmission(ethers.parseEther("10000"), 9900, 0))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");
        });

        it("should only allow admin to configure emission", async function () {
            await expect(
                autonet.connect(user1).configureEmission(ethers.parseEther("10000"), 9900, 86400)
            ).to.be.revertedWithCustomError(autonet, "NotAuthorized");
        });
    });

    describe("Auto-Epoch", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("auto-epoch-test"));

        beforeEach(async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
            await autonet.activateService(serviceId);
        });

        it("should advance epoch permissionlessly after duration elapses", async function () {
            await autonet.configureEmission(ethers.parseEther("10000"), 9900, 60); // 60s epochs

            expect(await autonet.currentEpoch()).to.equal(1);

            // Advance time past epoch duration
            await ethers.provider.send("evm_increaseTime", [61]);
            await ethers.provider.send("evm_mine", []);

            await expect(autonet.connect(user1).advanceEpoch())
                .to.emit(autonet, "AutoEpochAdvanced");

            expect(await autonet.currentEpoch()).to.equal(2);
            // Epoch 2 budget should be decayed
            expect(await autonet.epochBudget(2)).to.equal(ethers.parseEther("9900"));
        });

        it("should reject early epoch advance", async function () {
            await autonet.configureEmission(ethers.parseEther("10000"), 9900, 86400);

            await expect(autonet.connect(user1).advanceEpoch())
                .to.be.revertedWithCustomError(autonet, "EpochNotReady");
        });

        it("should auto-advance on attestUsage when epoch is ready", async function () {
            await autonet.configureEmission(ethers.parseEther("10000"), 9900, 60);

            expect(await autonet.currentEpoch()).to.equal(1);

            // Attest in epoch 1
            await autonet.connect(user1).attestUsage(serviceId, 50);

            // Advance time
            await ethers.provider.send("evm_increaseTime", [61]);
            await ethers.provider.send("evm_mine", []);

            // Next attestUsage should trigger auto-advance
            await autonet.connect(user1).attestUsage(serviceId, 25);

            expect(await autonet.currentEpoch()).to.equal(2);
            // The 25 units should be in epoch 2
            expect(await autonet.epochUsage(2, serviceId)).to.equal(25);
            // The 50 units should be finalized in epoch 1
            expect(await autonet.epochFinalized(1)).to.equal(true);
        });

        it("should finalize previous epoch on advance", async function () {
            await autonet.configureEmission(ethers.parseEther("10000"), 9900, 60);

            await autonet.connect(user1).attestUsage(serviceId, 100);

            await ethers.provider.send("evm_increaseTime", [61]);
            await ethers.provider.send("evm_mine", []);

            await autonet.connect(user1).advanceEpoch();

            // Epoch 1 should be finalized with rewards
            expect(await autonet.epochFinalized(1)).to.equal(true);
            const reward = await autonet.getServiceEpochReward(serviceId, 1);
            expect(reward.totalReward).to.equal(ethers.parseEther("10000"));
        });

        it("should reject advanceEpoch when auto-epoch is disabled", async function () {
            // Don't call configureEmission — auto-epoch not enabled
            await expect(autonet.connect(user1).advanceEpoch())
                .to.be.revertedWithCustomError(autonet, "NotAuthorized");
        });
    });

    describe("Participant Reward Claiming", function () {
        const serviceId = ethers.keccak256(ethers.toUtf8Bytes("claim-test"));
        const budget = ethers.parseEther("1000");

        beforeEach(async function () {
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
            await autonet.activateService(serviceId);
            await autonet.startEpoch(budget);
        });

        it("should allow participant to claim proportional reward", async function () {
            // user1 attests 75 units, user2 attests 25 units
            await autonet.connect(user1).attestUsage(serviceId, 75);
            await autonet.connect(user2).attestUsage(serviceId, 25);

            await autonet.finalizeCurrentEpoch();

            // user1 claims: 75/100 * 1000 = 750 ATN
            await expect(autonet.connect(user1).claimParticipantReward(serviceId, 1))
                .to.emit(autonet, "ParticipantRewardClaimed")
                .withArgs(serviceId, 1, user1.address, ethers.parseEther("750"));

            expect(await autonet.balanceOf(user1.address)).to.equal(ethers.parseEther("750"));
        });

        it("should allow multiple participants to claim", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 60);
            await autonet.connect(user2).attestUsage(serviceId, 40);

            await autonet.finalizeCurrentEpoch();

            await autonet.connect(user1).claimParticipantReward(serviceId, 1);
            await autonet.connect(user2).claimParticipantReward(serviceId, 1);

            expect(await autonet.balanceOf(user1.address)).to.equal(ethers.parseEther("600"));
            expect(await autonet.balanceOf(user2.address)).to.equal(ethers.parseEther("400"));
        });

        it("should prevent double-claiming", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 100);
            await autonet.finalizeCurrentEpoch();

            await autonet.connect(user1).claimParticipantReward(serviceId, 1);

            await expect(autonet.connect(user1).claimParticipantReward(serviceId, 1))
                .to.be.revertedWithCustomError(autonet, "AlreadyClaimed");
        });

        it("should reject claim before epoch is finalized", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 100);

            await expect(autonet.connect(user1).claimParticipantReward(serviceId, 1))
                .to.be.revertedWithCustomError(autonet, "EpochNotFinalized");
        });

        it("should reject claim from non-attester", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 100);
            await autonet.finalizeCurrentEpoch();

            // user2 never attested
            await expect(autonet.connect(user2).claimParticipantReward(serviceId, 1))
                .to.be.revertedWithCustomError(autonet, "NothingToClaim");
        });

        it("should track attester usage correctly", async function () {
            await autonet.connect(user1).attestUsage(serviceId, 30);
            await autonet.connect(user1).attestUsage(serviceId, 20);
            await autonet.connect(user2).attestUsage(serviceId, 50);

            expect(await autonet.getAttesterUsage(1, serviceId, user1.address)).to.equal(50);
            expect(await autonet.getAttesterUsage(1, serviceId, user2.address)).to.equal(50);
        });
    });

    describe("Admin Transfer", function () {
        it("should transfer admin role", async function () {
            await expect(autonet.transferAdmin(user1.address))
                .to.emit(autonet, "AdminTransferred")
                .withArgs(deployer.address, user1.address);

            expect(await autonet.admin()).to.equal(user1.address);
        });

        it("should not allow non-admin to transfer", async function () {
            await expect(autonet.connect(user1).transferAdmin(user2.address))
                .to.be.revertedWithCustomError(autonet, "NotAuthorized");
        });

        it("should not allow transfer to zero address", async function () {
            await expect(autonet.transferAdmin(ethers.ZeroAddress))
                .to.be.revertedWithCustomError(autonet, "ZeroAddress");
        });
    });

    describe("View Functions", function () {
        it("should list all service IDs", async function () {
            const id1 = ethers.keccak256(ethers.toUtf8Bytes("view-1"));
            const id2 = ethers.keccak256(ethers.toUtf8Bytes("view-2"));

            await autonet.connect(user1).registerService(id1, mockProject, "h1");
            await autonet.connect(user1).registerService(id2, mockProject, "h2");

            const allIds = await autonet.getAllServiceIds();
            expect(allIds.length).to.equal(2);
        });

        it("should list inference providers only", async function () {
            const generalId = ethers.keccak256(ethers.toUtf8Bytes("general"));
            const inferId = ethers.keccak256(ethers.toUtf8Bytes("inference"));

            await autonet.connect(user1).registerService(generalId, mockProject, "h1");
            await autonet.connect(user1).registerInferenceProvider(inferId, mockProject, user2.address, "h2");

            const providers = await autonet.getInferenceProviders();
            expect(providers.length).to.equal(1);
            expect(providers[0]).to.equal(inferId);
        });
    });

    describe("User Contracts", function () {
        const standardsHash = ethers.keccak256(ethers.toUtf8Bytes("jurisdiction-standards-v1"));

        it("should create user contract", async function () {
            await expect(autonet.connect(user1).createUserContract(standardsHash))
                .to.emit(autonet, "UserContractCreated")
                .withArgs(user1.address, (addr) => addr !== ethers.ZeroAddress);

            const userContractAddr = await autonet.getUserContract(user1.address);
            expect(userContractAddr).to.not.equal(ethers.ZeroAddress);

            // Verify user count
            expect(await autonet.getRegisteredUserCount()).to.equal(1);
        });

        it("should not allow duplicate user contracts", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);

            await expect(autonet.connect(user1).createUserContract(standardsHash))
                .to.be.revertedWithCustomError(autonet, "UserAlreadyRegistered");
        });

        it("should track registered users", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            await autonet.connect(user2).createUserContract(standardsHash);

            const users = await autonet.getRegisteredUsers();
            expect(users.length).to.equal(2);
            expect(users[0]).to.equal(user1.address);
            expect(users[1]).to.equal(user2.address);
        });

        it("should initialize user contract correctly", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            expect(await userContract.owner()).to.equal(user1.address);
            expect(await userContract.autonet()).to.equal(await autonet.getAddress());
            expect(await userContract.jurisdictionStandardsHash()).to.equal(standardsHash);
            expect(await userContract.alignmentScore()).to.equal(0);
        });

        it("should allow owner to set preferences", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            await expect(userContract.connect(user1).setPreference("theme", "dark"))
                .to.emit(userContract, "PreferenceSet")
                .withArgs("theme", "dark");

            expect(await userContract.getPreference("theme")).to.equal("dark");

            const keys = await userContract.getPreferenceKeys();
            expect(keys.length).to.equal(1);
            expect(keys[0]).to.equal("theme");
        });

        it("should not allow non-owner to set preferences", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            await expect(userContract.connect(user2).setPreference("theme", "dark"))
                .to.be.revertedWithCustomError(userContract, "NotOwner");
        });

        it("should allow owner to update alignment", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            await expect(userContract.connect(user1).updateAlignment(7500))
                .to.emit(userContract, "AlignmentUpdated")
                .withArgs(0, 7500);

            expect(await userContract.alignmentScore()).to.equal(7500);
        });

        it("should reject invalid alignment score", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            await expect(userContract.connect(user1).updateAlignment(10001))
                .to.be.revertedWithCustomError(userContract, "InvalidScore");
        });

        it("should update stats on usage attestation", async function () {
            // Create user contract
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            // Register and activate a service
            const serviceId = ethers.keccak256(ethers.toUtf8Bytes("user-stats-test"));
            await autonet.connect(user1).registerService(serviceId, mockProject, "hash");
            await autonet.activateService(serviceId);

            // Attest usage
            await autonet.connect(user1).attestUsage(serviceId, 100);
            await autonet.connect(user1).attestUsage(serviceId, 50);

            // Check stats
            expect(await userContract.totalAttestations()).to.equal(2);
            expect(await userContract.totalUsageUnits()).to.equal(150);
        });

        it("should only allow autonet to increment stats", async function () {
            await autonet.connect(user1).createUserContract(standardsHash);
            const userContractAddr = await autonet.getUserContract(user1.address);

            const AutonetUser = await ethers.getContractFactory("AutonetUser");
            const userContract = AutonetUser.attach(userContractAddr);

            // Direct call should fail
            await expect(userContract.connect(user1).incrementStats(100))
                .to.be.revertedWithCustomError(userContract, "NotAutonet");
        });
    });

    describe("Burn For Inference (Story 9.2)", function () {
        let mockRepTokenContract;

        beforeEach(async function () {
            // Get the RepToken contract used by this Autonet instance
            mockRepTokenContract = await ethers.getContractAt("AutonetMockRepToken", repToken);
        });

        // Helper: give user ATN via RepToken → convertGovernanceToAtn
        async function giveUserATN(user, amount) {
            // Mint RepTokens to user
            await mockRepTokenContract.mint(user.address, amount);
            // Approve Autonet to pull RepTokens
            await mockRepTokenContract.connect(user).approve(await autonet.getAddress(), amount);
            // Convert to ATN
            await autonet.connect(user).convertGovernanceToAtn(amount);
        }

        it("should burn ATN for tier 1 inference", async function () {
            const burnAmount = ethers.parseEther("10");
            await giveUserATN(user1, burnAmount);

            const balanceBefore = await autonet.balanceOf(user1.address);
            expect(balanceBefore).to.be.gte(burnAmount);

            await expect(autonet.connect(user1).burnForInference(burnAmount, 1))
                .to.emit(autonet, "InferenceBurned")
                .withArgs(user1.address, burnAmount, 1);

            const balanceAfter = await autonet.balanceOf(user1.address);
            expect(balanceAfter).to.equal(balanceBefore - burnAmount);
        });

        it("should burn ATN for tier 2 inference", async function () {
            const burnAmount = ethers.parseEther("50");
            await giveUserATN(user1, burnAmount);

            await expect(autonet.connect(user1).burnForInference(burnAmount, 2))
                .to.emit(autonet, "InferenceBurned")
                .withArgs(user1.address, burnAmount, 2);
        });

        it("should burn ATN for tier 3 inference", async function () {
            const burnAmount = ethers.parseEther("100");
            await giveUserATN(user1, burnAmount);

            await expect(autonet.connect(user1).burnForInference(burnAmount, 3))
                .to.emit(autonet, "InferenceBurned")
                .withArgs(user1.address, burnAmount, 3);
        });

        it("should track total inference burned", async function () {
            const burn1 = ethers.parseEther("10");
            const burn2 = ethers.parseEther("20");
            await giveUserATN(user1, burn1 + burn2);

            await autonet.connect(user1).burnForInference(burn1, 1);
            await autonet.connect(user1).burnForInference(burn2, 2);

            const stats = await autonet.getInferenceStats();
            expect(stats.totalBurned).to.equal(burn1 + burn2);
        });

        it("should track per-user inference burned", async function () {
            const burnAmount = ethers.parseEther("25");
            await giveUserATN(user1, burnAmount);

            await autonet.connect(user1).burnForInference(burnAmount, 1);
            expect(await autonet.userInferenceBurned(user1.address)).to.equal(burnAmount);
        });

        it("should track per-tier burn stats", async function () {
            const t1 = ethers.parseEther("10");
            const t2 = ethers.parseEther("20");
            const t3 = ethers.parseEther("30");
            await giveUserATN(user1, t1 + t2 + t3);

            await autonet.connect(user1).burnForInference(t1, 1);
            await autonet.connect(user1).burnForInference(t2, 2);
            await autonet.connect(user1).burnForInference(t3, 3);

            const tierStats = await autonet.getTierBurnStats();
            expect(tierStats.tier1Burned).to.equal(t1);
            expect(tierStats.tier2Burned).to.equal(t2);
            expect(tierStats.tier3Burned).to.equal(t3);
        });

        it("should reject zero amount", async function () {
            await expect(autonet.connect(user1).burnForInference(0, 1))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");
        });

        it("should reject invalid tier 0", async function () {
            const burnAmount = ethers.parseEther("10");
            await giveUserATN(user1, burnAmount);

            await expect(autonet.connect(user1).burnForInference(burnAmount, 0))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");
        });

        it("should reject invalid tier 4", async function () {
            const burnAmount = ethers.parseEther("10");
            await giveUserATN(user1, burnAmount);

            await expect(autonet.connect(user1).burnForInference(burnAmount, 4))
                .to.be.revertedWithCustomError(autonet, "InvalidRate");
        });

        it("should revert if user has insufficient balance", async function () {
            const burnAmount = ethers.parseEther("1000000");
            // user1 has no ATN
            await expect(autonet.connect(user1).burnForInference(burnAmount, 1))
                .to.be.reverted;
        });

        it("should reduce total supply on burn", async function () {
            const burnAmount = ethers.parseEther("50");
            await giveUserATN(user1, burnAmount);

            const supplyBefore = await autonet.totalSupply();
            await autonet.connect(user1).burnForInference(burnAmount, 2);
            const supplyAfter = await autonet.totalSupply();

            expect(supplyAfter).to.equal(supplyBefore - burnAmount);
        });
    });
});

describe("AutonetFactory", function () {
    let deployer, user1;
    let factory;
    let mockEconomy, mockRegistry, mockRepToken, mockTimelock;
    let mockRepTokenContract;

    beforeEach(async function () {
        [deployer, user1] = await ethers.getSigners();

        // Deploy mock contracts
        const MockRegistry = await ethers.getContractFactory("AutonetMockRegistry");
        const mockRegistryContract = await MockRegistry.deploy();

        const MockRepToken = await ethers.getContractFactory("AutonetMockRepToken");
        mockRepTokenContract = await MockRepToken.deploy("Test Rep", "REP");

        mockEconomy = deployer.address;
        mockRegistry = await mockRegistryContract.getAddress();
        mockRepToken = await mockRepTokenContract.getAddress();
        mockTimelock = deployer.address;

        // Give deployer some RepToken balance to allow deployment
        await mockRepTokenContract.mint(deployer.address, ethers.parseEther("100"));

        // Deploy implementation for factory reference (not actually used as clone)
        const Autonet = await ethers.getContractFactory("Autonet");
        const implementation = await Autonet.deploy(
            mockEconomy,
            mockRegistry,
            mockRepToken,
            mockTimelock,
            "Implementation",
            "IMPL"
        );

        // Deploy factory
        const AutonetFactory = await ethers.getContractFactory("AutonetFactory");
        factory = await AutonetFactory.deploy(await implementation.getAddress(), false);
    });

    it("should deploy factory correctly", async function () {
        expect(await factory.implementation()).to.not.equal(ethers.ZeroAddress);
        expect(await factory.enforceOnePerEconomy()).to.equal(false);
    });

    it("should create new Autonet instance", async function () {
        const tx = await factory.createAutonet(
            mockEconomy,
            mockRegistry,
            mockRepToken,
            mockTimelock,
            "Test ATN",
            "TATN"
        );

        const receipt = await tx.wait();
        const event = receipt.logs.find(log => log.eventName === "AutonetCreated");
        expect(event).to.not.be.undefined;

        const autonetAddress = event.args.autonet;
        const autonet = await ethers.getContractAt("Autonet", autonetAddress);

        expect(await autonet.name()).to.equal("Test ATN");
        expect(await autonet.symbol()).to.equal("TATN");
        expect(await autonet.admin()).to.equal(deployer.address); // transferred to deployer
    });

    it("should track deployments by deployer", async function () {
        await factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN1", "A1");
        await factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN2", "A2");

        const deployments = await factory.getAutonetsByDeployer(deployer.address);
        expect(deployments.length).to.equal(2);

        expect(await factory.getDeployedCount()).to.equal(2);
    });

    it("should enforce one-per-economy when enabled", async function () {
        await factory.setEnforcement(true);

        await factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN1", "A1");

        await expect(
            factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN2", "A2")
        ).to.be.revertedWithCustomError(factory, "AlreadyDeployedForEconomy");
    });

    it("should allow multiple per economy when not enforced", async function () {
        await factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN1", "A1");
        await factory.createAutonet(mockEconomy, mockRegistry, mockRepToken, mockTimelock, "ATN2", "A2");

        expect(await factory.getDeployedCount()).to.equal(2);
    });

    describe("Jurisdiction Ownership Verification", function () {
        it("should allow deployer with RepToken balance to deploy", async function () {
            // Deployer already has RepToken balance from beforeEach
            const tx = await factory.createAutonet(
                mockEconomy,
                mockRegistry,
                mockRepToken,
                mockTimelock,
                "Test ATN",
                "TATN"
            );

            const receipt = await tx.wait();
            const event = receipt.logs.find(log => log.eventName === "AutonetCreated");
            expect(event).to.not.be.undefined;

            const autonetAddress = event.args.autonet;
            expect(autonetAddress).to.not.equal(ethers.ZeroAddress);
        });

        it("should allow user with RepToken balance to deploy", async function () {
            // Give user1 RepToken balance
            await mockRepTokenContract.mint(user1.address, ethers.parseEther("50"));

            const tx = await factory.connect(user1).createAutonet(
                mockEconomy,
                mockRegistry,
                mockRepToken,
                mockTimelock,
                "User ATN",
                "UATN"
            );

            const receipt = await tx.wait();
            const event = receipt.logs.find(log => log.eventName === "AutonetCreated");
            expect(event).to.not.be.undefined;
            expect(event.args.deployer).to.equal(user1.address);
        });

        it("should reject deployer without RepToken balance", async function () {
            // user1 has no RepToken balance (no mint call)
            await expect(
                factory.connect(user1).createAutonet(
                    mockEconomy,
                    mockRegistry,
                    mockRepToken,
                    mockTimelock,
                    "Test ATN",
                    "TATN"
                )
            ).to.be.revertedWithCustomError(factory, "DeployerNotJurisdictionMember");
        });

        it("should reject deployer with zero RepToken balance after transfer", async function () {
            // Give user1 RepToken then transfer it all away
            await mockRepTokenContract.mint(user1.address, ethers.parseEther("100"));
            await mockRepTokenContract.connect(user1).transfer(deployer.address, ethers.parseEther("100"));

            // Now user1 has 0 balance
            expect(await mockRepTokenContract.balanceOf(user1.address)).to.equal(0);

            await expect(
                factory.connect(user1).createAutonet(
                    mockEconomy,
                    mockRegistry,
                    mockRepToken,
                    mockTimelock,
                    "Test ATN",
                    "TATN"
                )
            ).to.be.revertedWithCustomError(factory, "DeployerNotJurisdictionMember");
        });
    });
});
