// faq.js
module.exports = {
    matchFAQ: (msg) => {
        const lower = msg.toLowerCase();
        const faqs = {
            "return policy": "You can return most items within 90 days. Read more at walmart.com/help.",
            "order status": "Track your order at walmart.com/account/orders.",
            "store near me": "Please share your location so I can find the nearest Walmart.",
            "store hours": "Most Walmart stores are open from 6 AM to 11 PM."
        };
        const match = Object.keys(faqs).find(key => lower.includes(key));
        return match ? faqs[match] : null;
    }
};
