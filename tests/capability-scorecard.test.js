const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("CapabilityScorecard", function () {
    let deployer, evaluator1, evaluator2, user1;
    let scorecard;

    const VISUAL_ENCODER = ethers.keccak256(ethers.toUtf8Bytes("visual_encoder"));
    const TEXT_DECODER = ethers.keccak256(ethers.toUtf8Bytes("text_decoder"));
    const CROSS_MODAL = ethers.keccak256(ethers.toUtf8Bytes("cross_modal_fusion"));

    beforeEach(async function () {
        [deployer, evaluator1, evaluator2, user1] = await ethers.getSigners();

        const Scorecard = await ethers.getContractFactory("CapabilityScorecard");
        scorecard = await Scorecard.deploy(deployer.address);

        await scorecard.setEvaluator(evaluator1.address, true);
    });

    describe("Module Registration", function () {
        it("should register a module with target score", async function () {
            await expect(scorecard.registerModule(VISUAL_ENCODER, 8000))
                .to.emit(scorecard, "ModuleRegistered")
                .withArgs(VISUAL_ENCODER, 8000);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.targetScore).to.equal(8000);
            expect(mod.score).to.equal(0);
        });

        it("should not allow duplicate registration", async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
            await expect(scorecard.registerModule(VISUAL_ENCODER, 8000))
                .to.be.revertedWithCustomError(scorecard, "ModuleAlreadyExists");
        });

        it("should reject invalid target scores", async function () {
            await expect(scorecard.registerModule(VISUAL_ENCODER, 0))
                .to.be.revertedWithCustomError(scorecard, "InvalidScore");
            await expect(scorecard.registerModule(VISUAL_ENCODER, 10001))
                .to.be.revertedWithCustomError(scorecard, "InvalidScore");
        });

        it("should only allow admin to register", async function () {
            await expect(
                scorecard.connect(user1).registerModule(VISUAL_ENCODER, 8000)
            ).to.be.revertedWithCustomError(scorecard, "NotAuthorized");
        });
    });

    describe("Score Updates", function () {
        beforeEach(async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
        });

        it("should update score on first evaluation", async function () {
            await expect(scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 3000))
                .to.emit(scorecard, "ScoreUpdated")
                .withArgs(VISUAL_ENCODER, 0, 3000, evaluator1.address);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.score).to.equal(3000);
            expect(mod.evaluationCount).to.equal(1);
        });

        it("should use EMA for subsequent evaluations", async function () {
            // First: score = 4000
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            // Second: EMA = (4000 * 1 + 6000) / 2 = 5000
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 6000);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.score).to.equal(5000);
            expect(mod.evaluationCount).to.equal(2);
        });

        it("should reject updates from non-evaluators", async function () {
            await expect(
                scorecard.connect(user1).updateScore(VISUAL_ENCODER, 5000)
            ).to.be.revertedWithCustomError(scorecard, "NotAuthorized");
        });

        it("should reject score above max", async function () {
            await expect(
                scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 10001)
            ).to.be.revertedWithCustomError(scorecard, "InvalidScore");
        });

        it("should reject update for unregistered module", async function () {
            await expect(
                scorecard.connect(evaluator1).updateScore(TEXT_DECODER, 5000)
            ).to.be.revertedWithCustomError(scorecard, "ModuleNotFound");
        });
    });

    describe("Reward Multipliers", function () {
        beforeEach(async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
        });

        it("should return maxMultiplier (3x) for zero-score module", async function () {
            // Score is 0, target is 8000 → max multiplier
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(30000); // 3x
        });

        it("should return minMultiplier (0.5x) for saturated module", async function () {
            // Set score to target
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 8000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(5000); // 0.5x
        });

        it("should return minMultiplier for above-target score", async function () {
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 10000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(5000);
        });

        it("should interpolate for mid-range score", async function () {
            // Score = 4000, Target = 8000 → 50% of the way
            // Multiplier = 30000 - (50% * (30000 - 5000)) = 30000 - 12500 = 17500
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(17500); // 1.75x
        });

        it("should return base multiplier for unknown module", async function () {
            const unknown = ethers.keccak256(ethers.toUtf8Bytes("unknown"));
            const multiplier = await scorecard.getRewardMultiplier(unknown);
            expect(multiplier).to.equal(10000); // 1x base
        });
    });

    describe("Scorecard View", function () {
        it("should return full scorecard", async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
            await scorecard.registerModule(TEXT_DECODER, 6000);

            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            // TEXT_DECODER stays at 0

            const result = await scorecard.getScorecard();
            expect(result.ids.length).to.equal(2);
            expect(result.scores[0]).to.equal(4000);
            expect(result.scores[1]).to.equal(0);
            expect(result.targets[0]).to.equal(8000);
            expect(result.targets[1]).to.equal(6000);
            expect(result.multipliers[0]).to.equal(17500); // 1.75x
            expect(result.multipliers[1]).to.equal(30000); // 3x (no capability)
        });
    });

    describe("Evaluator Management", function () {
        it("should add and remove evaluators", async function () {
            await scorecard.setEvaluator(evaluator2.address, true);
            expect(await scorecard.evaluators(evaluator2.address)).to.equal(true);

            await scorecard.setEvaluator(evaluator2.address, false);
            expect(await scorecard.evaluators(evaluator2.address)).to.equal(false);
        });
    });
});

